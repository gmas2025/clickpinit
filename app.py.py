import streamlit as st
import os
from dotenv import load_dotenv
import requests
from urllib.parse import urlencode, urlparse, parse_qs
import base64
import shlex

# --- Load environment variables ---
load_dotenv()

# Pinterest API Credentials (using values from your .env)
CLIENT_ID = os.getenv("PINTEREST_CLIENT_ID")
CLIENT_SECRET = os.getenv("PINTEREST_CLIENT_SECRET")
# IMPORTANT: REDIRECT_URI MUST BE YOUR NGROK BASE URL (e.g., https://your-domain.ngrok-free.app/)
# And MUST be set exactly the same in your Pinterest Developer App settings.
REDIRECT_URI = os.getenv("REDIRECT_URI")

# AI API Keys (for image generation - used if you uncomment AI generation code)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Initialize AI Client (e.g., OpenAI DALL-E) ---
# Uncomment and ensure 'pip install openai' if you plan to use DALL-E.
# If you are using Gemini or another service, replace this initialization.
# openai_client = None
# if OPENAI_API_KEY:
#     try:
#         openai_client = OpenAI(api_key=OPENAI_API_KEY)
#     except Exception as e:
#         st.error(f"Could not initialize OpenAI client. Check OPENAI_API_KEY. Error: {e}")
# else:
#     st.warning("OPENAI_API_KEY not found in .env. Image generation functionality will be limited to placeholder.")

# --- Error Check for Essential Pinterest Environment Variables ---
if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
    st.error("❌ ERROR: Missing one or more essential Pinterest environment variables. Please check your .env file:")
    st.code(f"PINTEREST_CLIENT_ID: {CLIENT_ID}")
    st.code(f"PINTEREST_CLIENT_SECRET: {'*' * len(CLIENT_SECRET) if CLIENT_SECRET else 'None'}") # Mask secret for display
    st.code(f"REDIRECT_URI: {REDIRECT_URI}")
    st.stop() # Stop the app if essential variables are missing

# --- Build Pinterest Auth URL ---
def build_pinterest_auth_url():
    """Constructs the Pinterest OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI, # Now using the root URL as redirect URI
        "scope": "boards:read pins:write",
        "state": "secureRandomState123" # A random string to prevent CSRF attacks
    }
    return f"https://www.pinterest.com/oauth/?{urlencode(params)}"

# --- Image Generation Function (***PLACEHOLDER*** - REPLACE THIS WITH YOUR ACTUAL CODE) ---
def generate_image(prompt: str) -> str | None:
    """
    Generates an image based on a prompt.
    ***IMPORTANT: THIS IS CURRENTLY A PLACEHOLDER.***
    You need to replace the content of this function with your actual
    code for calling DALL-E, Gemini, or another image generation API.
    Ensure your implementation returns a publicly accessible URL to the image.
    """
    st.info(f"🎨 Attempting to generate image for prompt: '{prompt}'...")

    # --- DALL-E 3 Example (Uncomment and configure if using) ---
    # if openai_client:
    #     try:
    #         response = openai_client.images.generate(
    #             model="dall-e-3",
    #             prompt=prompt,
    #             size="1024x1024",
    #             quality="standard",
    #             n=1,
    #         )
    #         image_url = response.data[0].url
    #         st.success("✅ Image generated successfully with DALL-E 3!")
    #         st.image(image_url, caption="Generated Image Preview") # Display preview
    #         return image_url
    #     except Exception as e:
    #         st.error(f"❌ Failed to generate image with DALL-E 3: {e}")
    #         return None
    # else:
    #     st.warning("OpenAI client not initialized. Cannot use DALL-E for image generation.")

    # --- DEFAULT PLACEHOLDER (USED IF DALL-E CODE IS COMMENTED OUT/FAILS) ---
    st.warning("⚠️ Using a placeholder image URL. Replace 'generate_image' function with your actual AI image generation logic!")
    # This generates a unique random image from picsum.photos based on the prompt's hash
    return f"https://picsum.photos/1024/1024?random={hash(prompt) % 1000}"
    # --- END DEFAULT PLACEHOLDER ---

# --- Post any pin to Pinterest Production ---
def post_pin_to_pinterest(access_token: str, image_url: str, title: str, description: str, name: str, board_id: str):
    """
    Posts a pin with a given image URL to a specified Pinterest board in Pinterest.
    Requires an access token with 'pins:write' scope.
    """
    st.info(f"Attempting to post pin to board ID: {board_id}...")

    payload = {
        "board_id": board_id,
        "media_source": {"source_type": "image_url", "url": image_url},
        "title": title,
        "description": description,
        "name": name
    }

    try:
        post_resp = requests.post(
            "https://api.pinterest.com/v5/pins",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json=payload
        )
        post_resp.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        st.success("✅ Pin posted successfully to Pinterest!")
        st.json(post_resp.json())
        return True # Indicate success
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to post pin: {e}")
        st.text(f"Response status: {post_resp.status_code if 'post_resp' in locals() else 'N/A'}")
        st.text(f"Response text: {post_resp.text if 'post_resp' in locals() else 'No response available.'}")
        return False # Indicate failure

# --- Streamlit App Entry Point ---
def main():
    st.set_page_config(page_title="Pinterest OAuth + AI Image Generation", layout="wide")
    st.title("🚀 Pinterest OAuth + AI Generated Pin Poster")

    # Get query parameters using the standard st.query_params
    query_params = st.query_params
    auth_code = query_params.get("code") # st.query_params.get returns string or None directly
    state = query_params.get("state")

    # --- OAuth Authentication Flow ---
    # This block executes if an auth_code is present in the URL AND we don't have a token yet
    if auth_code and "pinterest_token" not in st.session_state:
        st.info("🔄 Authorization code received. Exchanging it for an access token...")

        # --- DEBUGGING: Display the received Authorization Code ---
        st.markdown("#### Debugging: Authorization Code Received")
        st.code(f"Received Authorization Code: {auth_code}")
        if auth_code == "2" or (auth_code and len(auth_code) < 10): # Check if the code is clearly bad
             st.error("⚠️ The received authorization code looks invalid or too short. This indicates Pinterest is not providing a valid code in the redirect.")
             st.warning("Please ensure your Pinterest Developer App's Redirect URI is *exactly* your ngrok URL (e.g., `https://moray-easy-mollusk.ngrok-free.app/oauth_callback/`) and matches the `REDIRECT_URI` in your `.env`.")
             st.stop() # Stop here if the code is clearly bad

        # Construct Basic Authorization header (Base64 encoded client_id:client_secret)
        client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(client_credentials.encode()).decode()
        auth_header = {"Authorization": f"Basic {encoded_credentials}"}

        # Prepare the token exchange payload
        token_payload_dict = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": REDIRECT_URI # Use the root URL as redirect URI for the exchange
        }
        token_payload_encoded = urlencode(token_payload_dict)

        # --- DEBUGGING: Display the curl command for manual testing ---
        curl_command = (
            f"curl -X POST 'https://api.pinterest.com/v5/oauth/token' \\\n"
            f"-H 'Content-Type: application/x-www-form-urlencoded' \\\n"
            f"-H 'Authorization: Basic {encoded_credentials}' \\\n"
            f"-d '{token_payload_encoded}'"
        )
        st.markdown("#### Manual cURL Command for Token Exchange:")
        st.code(curl_command)
        st.caption("Copy this command and run it in your terminal *immediately* after redirecting to verify the code validity.")

        try:
            with st.spinner("Exchanging code for access token..."):
                token_resp = requests.post(
                    "https://api.pinterest.com/v5/oauth/token",
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        **auth_header
                    },
                    data=token_payload_encoded
                )

            st.write("--- Pinterest Token Exchange Response ---")
            st.code(f"Status Code: {token_resp.status_code}")
            st.code(f"Response Text (raw): {token_resp.text}")

            token_resp.raise_for_status()

            token_data = token_resp.json()
            access_token = token_data.get("access_token")

            if access_token:
                st.session_state.pinterest_token = access_token
                st.success("✅ Successfully authenticated and retrieved Pinterest access token!")
                # Clear the URL query parameters after successful exchange
                st.query_params.clear() # Clears 'code' and 'state' from the URL
                st.rerun() # Forces a rerun with a clean URL (changed from experimental_rerun)
            else:
                st.error("❌ Access token not found in the response. Full response:")
                st.json(token_data)

        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error during token exchange: {e}")
            st.text(f"Response details: {token_resp.text}")
            st.warning("Please verify your Client ID, Client Secret, and Redirect URI in Pinterest Developer settings match exactly, and that your Basic Authentication is correctly formed. The 'authorization grant invalid' error usually means the received code is bad/expired.")
            return
        except requests.exceptions.ConnectionError as e:
            st.error(f"❌ Connection Error during token exchange: {e}")
            st.warning("Could be a network issue or ngrok tunnel problem. Check your internet connection and ngrok status.")
            return
        except requests.exceptions.JSONDecodeError as e:
            st.error(f"❌ Failed to decode JSON response from token exchange: {e}")
            st.text("Received non-JSON response from Pinterest token endpoint. Raw response:")
            st.code(token_resp.text)
            return
        except Exception as e:
            st.error(f"An unexpected error occurred during token exchange: {e}")
            return

    # --- Initial State / No Token: Show Login Link ---
    # This block executes if no auth_code is present in the URL AND no token in session state
    elif "pinterest_token" not in st.session_state:
        login_url = build_pinterest_auth_url() 
        st.write(f"DEBUG: Login URL: {login_url}")
        st.markdown(f"### [🔐 Click here to log in with Pinterest]({login_url})", unsafe_allow_html=True)
        st.info("Click the link above to start the OAuth authentication process and grant permissions to your app.")
        st.stop() # Stop further execution until redirect

    # --- Main Application UI (after successful authentication) ---
    access_token = st.session_state.get("pinterest_token")

    if access_token:
        st.success("🎉 You are authenticated with Pinterest. Ready to create and post pins!")

        # --- Board Selection UI ---
        st.subheader("Your Pinterest Boards (Pinterest Environment)")
        boards = []
        selected_board_name = None
        selected_board_id = None

        try:
            st.info("Fetching your Pinterest boards...")
            board_resp = requests.get(
                "https://api.pinterest.com/v5/boards",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            board_resp.raise_for_status()
            boards_data = board_resp.json().get("items", [])
            boards = sorted([(b.get("name"), b.get("id")) for b in boards_data if b.get("name") and b.get("id")], key=lambda x: x[0])

            if not boards:
                st.warning("No boards found in your Pinterest account. Please create one on the Pinterest Developer Dashboard (Pinterest) to enable posting.")
            else:
                board_name_to_id = {name: id_ for name, id_ in boards}
                board_names = [name for name, _ in boards]

                selected_board_name = st.selectbox("Select a Pinterest Board to post to:", board_names)
                selected_board_id = board_name_to_id.get(selected_board_name)
                st.info(f"Selected board: **{selected_board_name}** (ID: `{selected_board_id}`)")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to fetch boards from Pinterest: {e}")
            boards = []
        except Exception as e:
            st.error(f"An unexpected error occurred while fetching boards: {e}")

        # --- Pin Creation Sections (enabled only if a board is selected) ---
        if selected_board_id:
            st.markdown("---")
            st.subheader("🖼️ AI Image Generation and Pinterest Pin Creation")
            st.caption("❗**IMPORTANT**: The 'generate_image' function is currently a placeholder. Replace it with your actual DALL-E/Gemini code!")
            image_prompt = st.text_input(
                "Enter a prompt for AI image generation:",
                "A whimsical cat riding a bicycle through a field of giant sunflowers, watercolor style"
            )

            if st.button("✨ Generate Image & Post to Pinterest"):
                if not image_prompt:
                    st.warning("Please enter a prompt to generate an image.")
                    return

                generated_image_url = generate_image(image_prompt)

                if generated_image_url:
                    post_pin_to_pinterest(
                        access_token=access_token,
                        image_url=generated_image_url,
                        title=f"AI Pin: {image_prompt[:70]}{'...' if len(image_prompt) > 70 else ''}",
                        description=f"Generated by AI with prompt: '{image_prompt}'. Posted via Pinterest OAuth app.",
                        name="AI-Generated Pin",
                        board_id=selected_board_id
                    )
                else:
                    st.error("Could not obtain a valid image URL from generation. Pin not posted.")
            else:
                st.info("Enter a prompt above and click 'Generate Image & Post' to create an AI-powered pin.")

            st.markdown("---")

            st.subheader("📌 Post a Default Test Pin")
            st.info("This option posts a pre-defined image (Pleiades nebula) to Pinterest, using your selected board.")
            if st.button("Post Default Test Pin"):
                default_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Pleiades_large.jpg/800px-Pleiades_large.jpg"
                post_pin_to_pinterest(
                    access_token=access_token,
                    image_url=default_image_url,
                    title="Default Test Pin from OAuth App",
                    description="This pin uses a static image and was created after authenticating via Pinterest OAuth.",
                    name="OAuth Test Pin",
                    board_id=selected_board_id
                )
        else:
            if boards:
                st.info("Please select a board from the dropdown above to enable pin creation options.")

    else:
        # Fallback if somehow token is lost from session state after a successful exchange
        st.error("Pinterest Access Token is missing. Please restart the authentication process.")
        if st.button("Retry Pinterest Login"):
            st.session_state.clear()
            st.query_params.clear() # Ensure query params are also cleared
            st.rerun() # Changed from experimental_rerun

if __name__ == "__main__":
    main()