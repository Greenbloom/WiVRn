/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pose_list.h"
#include "driver/clock_offset.h"
#include "driver/polynomial_interpolator.h"
#include "driver/xrt_cast.h"
#include "math/m_api.h"
#include "math/m_eigen_interop.hpp"
#include "math/m_space.h"
#include "math/m_vec3.h"
#include "os/os_time.h"
#include "util/u_logging.h"
#include "xrt/xrt_defines.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <magic_enum.hpp>
#include <numbers>

using namespace xrt::auxiliary::math;

namespace wivrn
{
namespace
{
// How long a pose may go without a new sample before it is frozen. Longer than
// max extrapolation so a normal prediction horizon still uses the polynomial.
constexpr XrTime hold_stale_ns = 200'000'000;

// A wake-up sample this close to the held pose is real tracking, not a glitch.
constexpr float hold_close_distance_m = 0.40f;
constexpr float hold_close_angle_rad = 35.f * std::numbers::pi_v<float> / 180.f;

// Samples farther than that must agree with each other before they are trusted.
constexpr float hold_stable_distance_m = 0.03f;
constexpr float hold_stable_angle_rad = 10.f * std::numbers::pi_v<float> / 180.f;
constexpr int hold_stable_samples = 3;

// Quest often reports a tracked pose at the origin for a few frames after a
// controller wakes. Keep holding until that pose has been stable for a while.
constexpr float hold_origin_m = 0.05f;
constexpr float hold_origin_held_m = 0.20f;
constexpr int hold_origin_stable_samples = 8;
constexpr XrTime hold_origin_accept_ns = 300'000'000;

bool pose_is_tracked(uint8_t flags)
{
	constexpr uint8_t need = from_headset::pose_flags::orientation_valid |
	                         from_headset::pose_flags::position_valid |
	                         from_headset::pose_flags::orientation_tracked |
	                         from_headset::pose_flags::position_tracked;
	return (flags & need) == need;
}

float vec_length(const xrt_vec3 & v)
{
	return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float vec_distance(const xrt_vec3 & a, const xrt_vec3 & b)
{
	const float dx = a.x - b.x;
	const float dy = a.y - b.y;
	const float dz = a.z - b.z;
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float quat_angle(const xrt_quat & a, const xrt_quat & b)
{
	const float dot = std::fabs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
	return 2.f * std::acos(std::min(dot, 1.f));
}

bool notable_pose(device_id id)
{
	return id == device_id::LEFT_GRIP or id == device_id::RIGHT_GRIP or id == device_id::LEFT_AIM or id == device_id::RIGHT_AIM;
}

xrt_space_relation frozen_relation(const xrt_pose & pose)
{
	xrt_space_relation rel{};
	rel.relation_flags = xrt_space_relation_flags(
	        XRT_SPACE_RELATION_ORIENTATION_VALID_BIT |
	        XRT_SPACE_RELATION_POSITION_VALID_BIT |
	        XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT |
	        XRT_SPACE_RELATION_POSITION_TRACKED_BIT |
	        XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT |
	        XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT);
	rel.pose = pose;
	return rel;
}
} // namespace

xrt_space_relation pose_list::interpolate(const xrt_space_relation & a, const xrt_space_relation & b, float t)
{
	xrt_space_relation result;
	xrt_space_relation_flags flags = xrt_space_relation_flags(a.relation_flags & b.relation_flags);

	if (math_quat_dot(&a.pose.orientation, &b.pose.orientation) > 0)
	{
		m_space_relation_interpolate(const_cast<xrt_space_relation *>(&a), const_cast<xrt_space_relation *>(&b), t, flags, &result);
	}
	else
	{
		xrt_space_relation b2{
		        .relation_flags = b.relation_flags,
		        .pose = {
		                .orientation = {
		                        .x = -b.pose.orientation.x,
		                        .y = -b.pose.orientation.y,
		                        .z = -b.pose.orientation.z,
		                        .w = -b.pose.orientation.w,
		                },
		                .position = b.pose.position,
		        },
		        .linear_velocity = b.linear_velocity,
		        .angular_velocity = b.angular_velocity,
		};

		m_space_relation_interpolate(const_cast<xrt_space_relation *>(&a), &b2, t, flags, &result);
	}
	return result;
}

xrt_space_relation pose_list::extrapolate(const xrt_space_relation & a, const xrt_space_relation & b, int64_t ta, int64_t tb, int64_t t)
{
	float h = (tb - ta) / 1.e9;

	xrt_space_relation res = t < ta ? a : b;

	xrt_vec3 lin_vel = res.relation_flags & XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT ? res.linear_velocity : (b.pose.position - a.pose.position) / h;

	float dt = (t - tb) / 1.e9;

	float dt2_over_2 = dt * dt / 2;
	res.pose.position = res.pose.position + lin_vel * dt;

	if (res.relation_flags & XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT)
	{
		xrt_vec3 dtheta = res.angular_velocity * dt;
		xrt_quat dq;
		math_quat_exp(&dtheta, &dq);

		map_quat(res.pose.orientation) = map_quat(res.pose.orientation) * map_quat(dq);
	}

	return res;
}

pose_list::pose_list(wivrn::device_id id) : device(id)
{
	if (auto dump = std::getenv("WIVRN_DUMP"); dump and dump == std::string_view("list"))
		std::cerr << "WIVRN_DUMP_" << magic_enum::enum_name(id) << std::endl;
	if (auto dump = std::getenv(std::format("WIVRN_DUMP_{}", magic_enum::enum_name(id)).c_str()))
		dumper.emplace(dump);
}

std::pair<XrTime, XrTime> pose_list::get_bounds() const
{
	return positions.get_bounds();
}

void pose_list::update_tracking(const from_headset::tracking & tracking, const clock_offset & offset)
{
	if (source or not offset)
		return;

	for (const auto & pose: tracking.device_poses)
	{
		if (pose.device != device)
			continue;

		return add_sample(tracking.production_timestamp, tracking.timestamp, pose, offset);
	}
}

void pose_list::set_hold_when_untracked(bool enabled)
{
	hold_when_untracked = enabled;
}

void pose_list::set_derived(pose_list * source, xrt_pose offset, bool force)
{
	if (force)
	{
		derive_forced = true;
	}
	else if (derive_forced)
	{
		return;
	}
	// TODO: check for loops?
	if (source == this)
		this->source = nullptr;
	else
	{
		this->offset = offset;
		this->source = source;
	}
}

std::tuple<XrTime, xrt_space_relation, device_id> pose_list::get_pose_at(XrTime at_timestamp_ns)
{
	if (auto source = this->source.load())
	{
		auto res = source->get_pose_at(at_timestamp_ns);
		math_pose_transform(&std::get<1>(res).pose, &offset, &std::get<1>(res).pose);
		return res;
	}

	return std::tuple_cat(get_at(at_timestamp_ns), std::make_tuple(device));
}

void pose_list::reset()
{
	std::lock_guard lock(mutex);
	positions.reset();
	orientations.reset();
	// A profile switch clears the polynomial so the next pose does not blend
	// with the old one, but the last pose stays available as the hold anchor.
	if (hold_when_untracked and held.have)
	{
		held.holding = true;
		held.stable = 0;
	}
	else
		held = {};
}

bool pose_list::accept_tracked_sample(const from_headset::tracking::pose & pose, XrTime timestamp, bool & reset_history)
{
	reset_history = false;
	if (not pose_is_tracked(pose.flags))
	{
		if (not held.have)
			return true;

		if (not held.holding and notable_pose(device))
			U_LOG_I("Holding last %s pose", std::string(magic_enum::enum_name(device)).c_str());
		held.holding = true;
		held.stable = 0;
		return false;
	}

	const xrt_pose sample = xrt_cast(pose.pose);
	const bool resuming = held.have and (held.holding or timestamp - held.timestamp > hold_stale_ns);
	if (not resuming)
		return true;

	const float distance = vec_distance(sample.position, held.pose.position);
	const float angle = quat_angle(sample.orientation, held.pose.orientation);
	const bool near_origin = vec_length(sample.position) < hold_origin_m and vec_length(held.pose.position) > hold_origin_held_m;
	const bool close = distance < hold_close_distance_m and angle < hold_close_angle_rad and not near_origin;
	// Samples from before a long sleep make the polynomial fit unstable.
	if (close and timestamp - held.timestamp > 1'000'000'000)
		reset_history = true;
	if (not close)
	{
		const bool stable_with_candidate = held.stable > 0 and
		                                    vec_distance(sample.position, held.candidate.position) < hold_stable_distance_m and
		                                    quat_angle(sample.orientation, held.candidate.orientation) < hold_stable_angle_rad;
		if (stable_with_candidate)
			++held.stable;
		else
		{
			held.candidate = sample;
			held.stable = 1;
			held.candidate_since = timestamp;
		}

		const int needed = near_origin ? hold_origin_stable_samples : hold_stable_samples;
		const bool origin_ready = not near_origin or timestamp - held.candidate_since >= hold_origin_accept_ns;
		if (held.stable < needed or not origin_ready)
		{
			held.holding = true;
			return false;
		}
		reset_history = true;
	}

	if (held.holding and notable_pose(device))
		U_LOG_I("Resuming %s pose", std::string(magic_enum::enum_name(device)).c_str());
	held.holding = false;
	held.stable = 0;
	return true;
}

void pose_list::add_sample(XrTime production_timestamp, XrTime timestamp, const from_headset::tracking::pose & pose, const clock_offset & offset)
{
	production_timestamp = offset.from_headset(production_timestamp);
	timestamp = offset.from_headset(timestamp);

	// TODO keep the tracked flag

	polynomial_interpolator<3>::sample position{production_timestamp, timestamp};
	if (pose.flags & from_headset::pose_flags::position_valid)
		position.y.emplace(
		        pose.pose.position.x,
		        pose.pose.position.y,
		        pose.pose.position.z);
	if (pose.flags & from_headset::pose_flags::linear_velocity_valid)
		position.dy.emplace(
		        pose.linear_velocity.x,
		        pose.linear_velocity.y,
		        pose.linear_velocity.z);

	polynomial_interpolator<4, true>::sample orientation{production_timestamp, timestamp};
	if (pose.flags & from_headset::pose_flags::orientation_valid)
	{
		orientation.y.emplace(
		        pose.pose.orientation.w,
		        pose.pose.orientation.x,
		        pose.pose.orientation.y,
		        pose.pose.orientation.z);

		if (pose.flags & from_headset::pose_flags::angular_velocity_valid)
		{
			Eigen::Quaternionf q{
			        pose.pose.orientation.w,
			        pose.pose.orientation.x,
			        pose.pose.orientation.y,
			        pose.pose.orientation.z};

			Eigen::Quaternionf ω{
			        0,
			        pose.angular_velocity.x,
			        pose.angular_velocity.y,
			        pose.angular_velocity.z};

			orientation.dy.emplace(0.5 * (ω * q).coeffs());
		}
	}

	std::lock_guard lock(mutex);

	bool reset_history = false;
	if (hold_when_untracked and not accept_tracked_sample(pose, timestamp, reset_history))
		return;

	if (reset_history)
	{
		positions.reset();
		orientations.reset();
		// The first accepted sample after a jump often carries a garbage
		// velocity. Drop it so the hand does not flick.
		position.dy.reset();
		orientation.dy.reset();
	}

	positions.add_sample(position);
	orientations.add_sample(orientation);

	if (hold_when_untracked and pose_is_tracked(pose.flags))
	{
		held.have = true;
		held.holding = false;
		held.stable = 0;
		held.timestamp = timestamp;
		held.pose = xrt_cast(pose.pose);
	}

	if (dumper)
	{
		debug_data item{
		        .in = true,
		        .production_timestamp = production_timestamp,
		        .timestamp = timestamp,
		        .now = os_monotonic_get_ns(),
		};
		if (position.y)
			Eigen::Map<Eigen::Vector<float, 3>>(item.position.data()) = *position.y;
		if (position.dy)
			Eigen::Map<Eigen::Vector<float, 3>>(item.dposition.data()) = *position.dy;
		if (orientation.y)
			Eigen::Map<Eigen::Vector<float, 4>>(item.orientation.data()) = *orientation.y;
		if (orientation.dy)
			Eigen::Map<Eigen::Vector<float, 4>>(item.dorientation.data()) = *orientation.dy;
		dumper->write(item);
	}
}

std::pair<XrTime, xrt_space_relation> pose_list::get_at(XrTime at_timestamp_ns)
{
	std::lock_guard lock(mutex);

	if (hold_when_untracked and held.have and (held.holding or at_timestamp_ns - held.timestamp > hold_stale_ns))
	{
		// Report "now" so the tracking feedback loop does not treat the
		// frozen sample as seconds of motion-to-photon latency.
		return {os_monotonic_get_ns(), frozen_relation(held.pose)};
	}

	xrt_space_relation ret{};
	auto flags = [&](int f) {
		ret.relation_flags = xrt_space_relation_flags(int(ret.relation_flags) | f);
	};

	auto position = positions.get_at(at_timestamp_ns);
	auto orientation = orientations.get_at(at_timestamp_ns);

	if (position.y)
	{
		flags(XRT_SPACE_RELATION_POSITION_VALID_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);
		ret.pose.position = {
		        position.y->x(),
		        position.y->y(),
		        position.y->z()};
	}

	if (position.dy)
	{
		flags(XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT);
		ret.linear_velocity = {
		        position.dy->x(),
		        position.dy->y(),
		        position.dy->z()};
	}

	if (orientation.y)
	{
		if (auto norm2 = orientation.y->squaredNorm(); norm2 > 0.1)
		{
			flags(XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT);

			orientation.y->normalize();

			ret.pose.orientation = {
			        .x = (*orientation.y)[1],
			        .y = (*orientation.y)[2],
			        .z = (*orientation.y)[3],
			        .w = (*orientation.y)[0]};
		}

		if (orientation.dy)
		{
			Eigen::Quaternionf q{
			        (*orientation.y)[0],
			        (*orientation.y)[1],
			        (*orientation.y)[2],
			        (*orientation.y)[3]};
			Eigen::Quaternionf dq{
			        (*orientation.dy)[0],
			        (*orientation.dy)[1],
			        (*orientation.dy)[2],
			        (*orientation.dy)[3]};
			Eigen::Quaternionf half_ω = dq * q.conjugate();

			flags(XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT);
			ret.angular_velocity = {
			        2 * half_ω.x(),
			        2 * half_ω.y(),
			        2 * half_ω.z(),
			};
		}
	}

	if (dumper)
	{
		debug_data item{
		        .in = false,
		        .production_timestamp = position.production_timestamp,
		        .timestamp = at_timestamp_ns,
		        .now = os_monotonic_get_ns(),
		};
		if (position.y)
			Eigen::Map<Eigen::Vector<float, 3>>(item.position.data()) = *position.y;
		if (position.dy)
			Eigen::Map<Eigen::Vector<float, 3>>(item.dposition.data()) = *position.dy;
		if (orientation.y)
			Eigen::Map<Eigen::Vector<float, 4>>(item.orientation.data()) = *orientation.y;
		if (orientation.dy)
			Eigen::Map<Eigen::Vector<float, 4>>(item.dorientation.data()) = *orientation.dy;
		dumper->write(item);
	}

	return {position.production_timestamp, ret};
}

} // namespace wivrn
