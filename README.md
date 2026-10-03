# NovaMart AI Support

## Run locally

Install dependencies and start the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a local `.env` file with `APP_USERNAME`, `APP_PASSWORD`, and the API key
for the provider you want to use (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, or
`ANTHROPIC_API_KEY`). Do not commit `.env`.

## Deploy to Streamlit Community Cloud

Create an app from this repository's `main` branch and select `app.py` as the
entry point. In the app's **Settings → Secrets**, configure:

```toml
APP_USERNAME = "choose-a-private-username"
APP_PASSWORD = "choose-a-strong-password"
OPENAI_API_KEY = "your-rotated-openai-api-key"
```

Use a newly issued API key; never put API keys in Git or share them in chat.
The app login credentials and provider key are read from Streamlit secrets
when deployed.
