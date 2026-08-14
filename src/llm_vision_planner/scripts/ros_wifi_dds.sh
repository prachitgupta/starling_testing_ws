#!/usr/bin/env bash
# Source this file so its DDS environment applies to commands in the current shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this script; do not execute it." >&2
  echo "Usage: source $0 {enable WIFI_INTERFACE|auto [DOMAIN_ID]|disable|status}" >&2
  exit 2
fi

_ros_wifi_dds_restore_var() {
  local name="$1"
  local was_set="$2"
  local old_value="$3"
  if [[ "$was_set" == "yes" ]]; then
    export "$name=$old_value"
  else
    unset "$name"
  fi
}

case "${1:-}" in
  enable)
    _ros_wifi_dds_interface="${2:-}"
    _ros_wifi_dds_domain="${3:-${ROS_DOMAIN_ID:-0}}"

    if [[ -z "$_ros_wifi_dds_interface" ]]; then
      echo "Usage: source ${BASH_SOURCE[0]} enable WIFI_INTERFACE|auto [DOMAIN_ID]" >&2
      return 2
    fi
    if ! [[ "$_ros_wifi_dds_domain" =~ ^[0-9]+$ ]]; then
      echo "ERROR: ROS domain ID must be a non-negative integer." >&2
      return 2
    fi
    if [[ "$_ros_wifi_dds_interface" == "auto" ]]; then
      # Prefer an up wireless interface with a global IPv4 address (the
      # flight Wi-Fi) over the default route, which may point at a wired
      # uplink on a dual-homed ground station.
      _ros_wifi_dds_detected=""
      for _ros_wifi_dds_candidate in /sys/class/net/*/wireless; do
        [[ -d "$_ros_wifi_dds_candidate" ]] || continue
        _ros_wifi_dds_candidate="$(basename "$(dirname "$_ros_wifi_dds_candidate")")"
        if ip -o -4 address show dev "$_ros_wifi_dds_candidate" scope global \
             up >/dev/null 2>&1; then
          _ros_wifi_dds_detected="$_ros_wifi_dds_candidate"
          break
        fi
      done
      unset _ros_wifi_dds_candidate
      if [[ -z "$_ros_wifi_dds_detected" ]]; then
        _ros_wifi_dds_detected="$(
          ip -o -4 route show default \
            | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}'
        )"
      fi
      if [[ -z "$_ros_wifi_dds_detected" ]]; then
        echo "ERROR: could not auto-detect a wireless interface or default route." >&2
        echo "Pass the interface name explicitly instead of 'auto'." >&2
        return 1
      fi
      echo "Auto-detected interface: ${_ros_wifi_dds_detected}"
      _ros_wifi_dds_interface="$_ros_wifi_dds_detected"
      unset _ros_wifi_dds_detected
    fi
    if ! ip link show dev "$_ros_wifi_dds_interface" >/dev/null 2>&1; then
      echo "ERROR: interface '$_ros_wifi_dds_interface' does not exist." >&2
      return 1
    fi

    _ros_wifi_dds_ipv4="$(
      ip -4 -o address show dev "$_ros_wifi_dds_interface" scope global \
        | awk 'NR == 1 {split($4, address, "/"); print address[1]}'
    )"
    if [[ -z "$_ros_wifi_dds_ipv4" ]]; then
      echo "ERROR: interface '$_ros_wifi_dds_interface' has no global IPv4 address." >&2
      return 1
    fi

    _ros_wifi_dds_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    _ros_wifi_dds_template="${_ros_wifi_dds_script_dir}/../config/fastdds_wifi_only.xml.in"
    if [[ ! -f "$_ros_wifi_dds_template" ]] && command -v ros2 >/dev/null 2>&1; then
      _ros_wifi_dds_prefix="$(ros2 pkg prefix llm_vision_planner 2>/dev/null || true)"
      _ros_wifi_dds_template="${_ros_wifi_dds_prefix}/share/llm_vision_planner/config/fastdds_wifi_only.xml.in"
    fi
    if [[ ! -f "$_ros_wifi_dds_template" ]]; then
      echo "ERROR: cannot find config/fastdds_wifi_only.xml.in." >&2
      return 1
    fi
    if command -v ros2 >/dev/null 2>&1 \
      && ! ros2 pkg prefix rmw_fastrtps_cpp >/dev/null 2>&1; then
      echo "ERROR: rmw_fastrtps_cpp is not installed in the sourced ROS environment." >&2
      return 1
    fi

    if [[ "${__ROS_WIFI_DDS_ACTIVE:-no}" != "yes" ]]; then
      __ROS_WIFI_DDS_OLD_RMW_SET="${RMW_IMPLEMENTATION+x}"
      __ROS_WIFI_DDS_OLD_RMW="${RMW_IMPLEMENTATION-}"
      __ROS_WIFI_DDS_OLD_FASTRTPS_SET="${FASTRTPS_DEFAULT_PROFILES_FILE+x}"
      __ROS_WIFI_DDS_OLD_FASTRTPS="${FASTRTPS_DEFAULT_PROFILES_FILE-}"
      __ROS_WIFI_DDS_OLD_FASTDDS_SET="${FASTDDS_DEFAULT_PROFILES_FILE+x}"
      __ROS_WIFI_DDS_OLD_FASTDDS="${FASTDDS_DEFAULT_PROFILES_FILE-}"
      __ROS_WIFI_DDS_OLD_DOMAIN_SET="${ROS_DOMAIN_ID+x}"
      __ROS_WIFI_DDS_OLD_DOMAIN="${ROS_DOMAIN_ID-}"
      __ROS_WIFI_DDS_OLD_LOCALHOST_SET="${ROS_LOCALHOST_ONLY+x}"
      __ROS_WIFI_DDS_OLD_LOCALHOST="${ROS_LOCALHOST_ONLY-}"
    fi

    _ros_wifi_dds_runtime_dir="${XDG_RUNTIME_DIR:-/tmp}/llm_vision_planner"
    mkdir -p -- "$_ros_wifi_dds_runtime_dir"
    chmod 700 "$_ros_wifi_dds_runtime_dir"
    _ros_wifi_dds_safe_interface="${_ros_wifi_dds_interface//[^[:alnum:]_.-]/_}"
    __ROS_WIFI_DDS_RUNTIME_FILE="${_ros_wifi_dds_runtime_dir}/fastdds-${_ros_wifi_dds_safe_interface}-${BASHPID}.xml"
    sed "s/@ROS_WIFI_IPV4@/${_ros_wifi_dds_ipv4}/g" \
      "$_ros_wifi_dds_template" >"$__ROS_WIFI_DDS_RUNTIME_FILE"
    chmod 600 "$__ROS_WIFI_DDS_RUNTIME_FILE"

    export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
    # Foxy/Humble Fast DDS uses FASTRTPS_*. Newer releases use FASTDDS_*.
    export FASTRTPS_DEFAULT_PROFILES_FILE="$__ROS_WIFI_DDS_RUNTIME_FILE"
    export FASTDDS_DEFAULT_PROFILES_FILE="$__ROS_WIFI_DDS_RUNTIME_FILE"
    export ROS_DOMAIN_ID="$_ros_wifi_dds_domain"
    export ROS_LOCALHOST_ONLY="0"
    __ROS_WIFI_DDS_ACTIVE="yes"
    __ROS_WIFI_DDS_INTERFACE="$_ros_wifi_dds_interface"
    __ROS_WIFI_DDS_IPV4="$_ros_wifi_dds_ipv4"

    if command -v ros2 >/dev/null 2>&1; then
      ros2 daemon stop >/dev/null 2>&1 || true
    fi
    echo "Fast DDS restricted to ${__ROS_WIFI_DDS_INTERFACE} (${__ROS_WIFI_DDS_IPV4}), domain ${ROS_DOMAIN_ID}."
    echo "Restart ROS 2 processes launched before this profile was enabled."
    ;;

  disable)
    if [[ "${__ROS_WIFI_DDS_ACTIVE:-no}" != "yes" ]]; then
      echo "ROS Wi-Fi DDS profile is not active in this shell."
      return 0
    fi

    _ros_wifi_dds_restore_var RMW_IMPLEMENTATION \
      "${__ROS_WIFI_DDS_OLD_RMW_SET:+yes}" "${__ROS_WIFI_DDS_OLD_RMW-}"
    _ros_wifi_dds_restore_var FASTRTPS_DEFAULT_PROFILES_FILE \
      "${__ROS_WIFI_DDS_OLD_FASTRTPS_SET:+yes}" "${__ROS_WIFI_DDS_OLD_FASTRTPS-}"
    _ros_wifi_dds_restore_var FASTDDS_DEFAULT_PROFILES_FILE \
      "${__ROS_WIFI_DDS_OLD_FASTDDS_SET:+yes}" "${__ROS_WIFI_DDS_OLD_FASTDDS-}"
    _ros_wifi_dds_restore_var ROS_DOMAIN_ID \
      "${__ROS_WIFI_DDS_OLD_DOMAIN_SET:+yes}" "${__ROS_WIFI_DDS_OLD_DOMAIN-}"
    _ros_wifi_dds_restore_var ROS_LOCALHOST_ONLY \
      "${__ROS_WIFI_DDS_OLD_LOCALHOST_SET:+yes}" "${__ROS_WIFI_DDS_OLD_LOCALHOST-}"

    rm -f -- "${__ROS_WIFI_DDS_RUNTIME_FILE:?}"
    unset __ROS_WIFI_DDS_ACTIVE __ROS_WIFI_DDS_INTERFACE __ROS_WIFI_DDS_IPV4
    unset __ROS_WIFI_DDS_RUNTIME_FILE
    unset __ROS_WIFI_DDS_OLD_RMW_SET __ROS_WIFI_DDS_OLD_RMW
    unset __ROS_WIFI_DDS_OLD_FASTRTPS_SET __ROS_WIFI_DDS_OLD_FASTRTPS
    unset __ROS_WIFI_DDS_OLD_FASTDDS_SET __ROS_WIFI_DDS_OLD_FASTDDS
    unset __ROS_WIFI_DDS_OLD_DOMAIN_SET __ROS_WIFI_DDS_OLD_DOMAIN
    unset __ROS_WIFI_DDS_OLD_LOCALHOST_SET __ROS_WIFI_DDS_OLD_LOCALHOST
    if command -v ros2 >/dev/null 2>&1; then
      ros2 daemon stop >/dev/null 2>&1 || true
    fi
    echo "Restored the DDS environment that this shell had before enable."
    ;;

  status)
    if [[ "${__ROS_WIFI_DDS_ACTIVE:-no}" == "yes" ]]; then
      echo "active: yes"
      echo "interface: ${__ROS_WIFI_DDS_INTERFACE}"
      echo "IPv4: ${__ROS_WIFI_DDS_IPV4}"
      echo "domain: ${ROS_DOMAIN_ID}"
      echo "profile: ${FASTRTPS_DEFAULT_PROFILES_FILE}"
    else
      echo "active: no"
    fi
    ;;

  *)
    echo "Usage: source ${BASH_SOURCE[0]} {enable WIFI_INTERFACE|auto [DOMAIN_ID]|disable|status}" >&2
    return 2
    ;;
esac

unset _ros_wifi_dds_interface _ros_wifi_dds_domain _ros_wifi_dds_ipv4
unset _ros_wifi_dds_script_dir _ros_wifi_dds_template _ros_wifi_dds_prefix
unset _ros_wifi_dds_runtime_dir _ros_wifi_dds_safe_interface
unset -f _ros_wifi_dds_restore_var
