import time

import requests

BASE_URL = "https://api.openf1.org/v1"

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5

DEFAULT_RETRY_SECONDS = 2.0
MAX_RETRY_SECONDS = 60.0


class OpenF1Client:
    """Simple OpenF1 API client.

    Retries 429, 5xx and network errors; does not retry 404.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
    ) -> None:
        self.base_url = base_url.rstrip("/")

        self.session = requests.Session()

    def get(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> list:
        """Fetch an OpenF1 endpoint.

        429 and 5xx are retried; 404 and other errors raise HTTPError immediately.
        """

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                response = self.session.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

            except requests.RequestException as exc:

                if attempt == MAX_RETRIES:
                    raise

                wait_seconds = min(
                    DEFAULT_RETRY_SECONDS * (2 ** (attempt - 1)),
                    MAX_RETRY_SECONDS,
                )

                print(
                    f"Request failed: {exc}"
                )

                print(
                    f"Retrying in {wait_seconds:.1f}s..."
                )

                time.sleep(wait_seconds)

                continue

            # --- SUCCESS ---

            if response.status_code == 200:
                return response.json()

            # --- 404 ---
            # Not transient (e.g. no data for an older race), so no retry.

            if response.status_code == 404:
                response.raise_for_status()

            # --- 429 - RATE LIMIT ---

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:

                    try:
                        wait_seconds = float(
                            retry_after
                        )

                    except ValueError:
                        wait_seconds = (
                            DEFAULT_RETRY_SECONDS
                        )

                else:

                    wait_seconds = min(
                        DEFAULT_RETRY_SECONDS * (2 ** (attempt - 1)),
                        MAX_RETRY_SECONDS,
                    )

                print(
                    f"Rate limit reached. "
                    f"Waiting {wait_seconds:.1f}s..."
                )

                time.sleep(wait_seconds)

                continue

            # --- 5xx - SERVER ERROR ---

            if 500 <= response.status_code < 600:

                if attempt == MAX_RETRIES:
                    response.raise_for_status()

                wait_seconds = min(
                    DEFAULT_RETRY_SECONDS * (2 ** (attempt - 1)),
                    MAX_RETRY_SECONDS,
                )

                print(
                    f"Server error "
                    f"{response.status_code}."
                )

                print(
                    f"Retrying in {wait_seconds:.1f}s..."
                )

                time.sleep(wait_seconds)

                continue

            # --- OTHER HTTP ERROR ---

            response.raise_for_status()

        raise RuntimeError(
            f"Failed to fetch {url} "
            f"after {MAX_RETRIES} attempts."
        )