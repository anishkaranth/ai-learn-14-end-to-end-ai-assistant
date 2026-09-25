# API rate limits
The REST API allows 60 requests per minute on the Free plan and 600 requests per minute on paid plans.
When you exceed the limit the API returns HTTP 429 with a Retry-After header in seconds.
API keys are created under Settings, Developer, and can be scoped to read-only access.
Rotate keys at least every 90 days; revoked keys stop working immediately.
