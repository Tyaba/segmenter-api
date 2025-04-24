import google.auth.transport.requests
import google.oauth2.id_token


def get_bearer_token(audience_url: str) -> str:
    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(
        request=auth_req, audience=audience_url
    )
    assert isinstance(token, str)
    return token


def get_authorized_headers(audience_url: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {get_bearer_token(audience_url)}"}
