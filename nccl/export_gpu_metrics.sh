#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <sqlite_file> [start_timestamp_ms] [end_timestamp_ms]" >&2
  exit 1
fi

sqlite_file="$1"
start_timestamp_ms="${2:-}"
end_timestamp_ms="${3:-}"

timestamp_filter=""
if [ -n "$start_timestamp_ms" ]; then
  timestamp_filter+=" AND g.timestamp / 1000000.0 >= $start_timestamp_ms"
fi
if [ -n "$end_timestamp_ms" ]; then
  timestamp_filter+=" AND g.timestamp / 1000000.0 <= $end_timestamp_ms"
fi

sqlite3 -header -csv "$sqlite_file" "
WITH base AS (
  SELECT
    g.timestamp / 1000000.0 AS timestamp_ms,
    t.metricName,
    g.value AS metric_value
  FROM GPU_METRICS g
  JOIN TARGET_INFO_GPU_METRICS t
    ON g.metricId = t.metricId
  WHERE t.metricName IN (
    'DRAM Read Bandwidth [Throughput %]',
    'DRAM Write Bandwidth [Throughput %]',
    'Tensor Active [Throughput %]',
    'Compute Warps in Flight [Throughput %]',
    'NVLink RX Requests Protocol Data [Throughput %]',
    'NVLink RX Requests User Data [Throughput %]',
    'NVLink RX Responses Protocol Data [Throughput %]',
    'NVLink RX Responses User Data [Throughput %]',
    'NVLink TX Requests Protocol Data [Throughput %]',
    'NVLink TX Requests User Data [Throughput %]',
    'NVLink TX Responses Protocol Data [Throughput %]',
    'NVLink TX Responses User Data [Throughput %]'
  )
  $timestamp_filter
)
SELECT
  timestamp_ms,
  MAX(CASE WHEN metricName='DRAM Read Bandwidth [Throughput %]'
           THEN metric_value END) AS dram_read_throughput_pct,
  MAX(CASE WHEN metricName='DRAM Write Bandwidth [Throughput %]'
           THEN metric_value END) AS dram_write_throughput_pct,
  MAX(CASE WHEN metricName='Tensor Active [Throughput %]'
           THEN metric_value END) AS tensor_active_throughput_pct,
  MAX(CASE WHEN metricName='Compute Warps in Flight [Throughput %]'
           THEN metric_value END) AS compute_warps_in_flight_throughput_pct,

  MAX(CASE WHEN metricName='NVLink RX Requests Protocol Data [Throughput %]'
           THEN metric_value END) AS nvlink_rx_req_protocol_pct,
  MAX(CASE WHEN metricName='NVLink RX Requests User Data [Throughput %]'
           THEN metric_value END) AS nvlink_rx_req_user_pct,
  MAX(CASE WHEN metricName='NVLink RX Responses Protocol Data [Throughput %]'
           THEN metric_value END) AS nvlink_rx_resp_protocol_pct,
  MAX(CASE WHEN metricName='NVLink RX Responses User Data [Throughput %]'
           THEN metric_value END) AS nvlink_rx_resp_user_pct,

  MAX(CASE WHEN metricName='NVLink TX Requests Protocol Data [Throughput %]'
           THEN metric_value END) AS nvlink_tx_req_protocol_pct,
  MAX(CASE WHEN metricName='NVLink TX Requests User Data [Throughput %]'
           THEN metric_value END) AS nvlink_tx_req_user_pct,
  MAX(CASE WHEN metricName='NVLink TX Responses Protocol Data [Throughput %]'
           THEN metric_value END) AS nvlink_tx_resp_protocol_pct,
  MAX(CASE WHEN metricName='NVLink TX Responses User Data [Throughput %]'
           THEN metric_value END) AS nvlink_tx_resp_user_pct

FROM base
GROUP BY timestamp_ms
ORDER BY timestamp_ms;
" > gpu_metrics_wide.csv
