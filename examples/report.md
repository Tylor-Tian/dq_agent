# Summary
| Key | Value |
| --- | --- |
| Run ID | 1280de6ea2e0462aaecfc607f6ff6a2d |
| Rows | 500 |
| Cols | 5 |
| Contract Errors | 0 |
| Contract Warnings | 0 |
| Contract Info | 0 |

# Rule Results
- unique:order_id (order_id): FAIL
- not_null:user_id (user_id): PASS
- range:amount (amount): FAIL
- allowed_values:status (status): FAIL

# Anomalies
- missing_rate (user_id): FAIL
- outlier_mad (amount): FAIL
