import streamlit as st
import os
# from dotenv import load_dotenv # No longer strictly needed if using st.secrets exclusively
import google.generativeai as genai
import json
# import json5 # Not used in provided code, keeping it commented
import pandas as pd
from openai import OpenAI
import requests
from PIL import Image
import io
from google.cloud import storage
import time
import re
import urllib.parse # For URL encoding OAuth parameters
import uuid # For generating unique state for OAuth CSRF protection

# --- Configuration & Initialization ---

# No need for load_dotenv() if relying entirely on Streamlit Cloud Secrets
# For local development, you'd place these in a .streamlit/secrets.toml file:
# [secrets]
# GEMINI_API_KEY = "your_gemini_api_key"
# OPENAI_API_KEY = "your_openai_api_key"
# GCP_PROJECT_ID = "your_gcp_project_id"
# PINTEREST_CLIENT_ID = "your_pinterest_app_id"
# PINTEREST_CLIENT_SECRET = "your_pinterest_app_secret"
# REDIRECT_URI = "http://localhost:8501" # Or your deployed Streamlit Cloud URL (e.g., https://your-app-name.streamlit.app)


# Configure API Keys and Secrets using st.secrets
# Use .get() or try-except for robust access and clearer error messages
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GCP_PROJECT_ID = st.secrets.get("GCP_PROJECT_ID")

# Pinterest OAuth Secrets
PINTEREST_CLIENT_ID = st.secrets.get("PINTEREST_CLIENT_ID")
PINTEREST_CLIENT_SECRET = st.secrets.get("PINTEREST_CLIENT_SECRET")
REDIRECT_URI = st.secrets.get("REDIRECT_URI") # This MUST match your Pinterest Developer App setting

# Define GCS buckets
GCS_IMAGES_BUCKET = "seo-pinterest-app-images-malko"
GCS_DATA_BUCKET = "seo-pinterest-app-images-malko"
HISTORY_BLOB_NAME = "data/pinterest_pin_history.jsonl"

# Define the structure for the DataFrame (columns and their order)
TABLE_HEADERS = [
    "Title", "Subtitle", "Hook", "Image Background", "Description",
    "Hashtags", "Alt Text", "Image_Url", "Pinterest URL", "Status", "Board Name", "Pinterest Pin ID"
]

# Define Pinterest API Endpoints and Scopes
PINTEREST_AUTH_URL = "https://www.pinterest.com/oauth/"
PINTEREST_TOKEN_URL = "https://api.pinterest.com/v5/oauth/token"
# IMPORTANT: Adjust scopes based on what your app actually needs to do on Pinterest
PINTEREST_SCOPES = "boards:read,pins:read,pins:write" # Example: Read boards, read pins, write pins

# --- 1. Initial Secret Checks (Critical for deployment) ---
# These checks prevent the app from even starting if essential secrets are missing.

if not GEMINI_API_KEY:
    st.error("❌ ERROR: GEMINI_API_KEY not found in Streamlit secrets. Please add it to your `.streamlit/secrets.toml` or Streamlit Cloud secrets.")
    st.stop()
if not OPENAI_API_KEY:
    st.error("❌ ERROR: OPENAI_API_KEY not found in Streamlit secrets. Please add it to your `.streamlit/secrets.toml` or Streamlit Cloud secrets.")
    st.stop()
if not PINTEREST_CLIENT_ID:
    st.error("❌ ERROR: PINTEREST_CLIENT_ID not found in Streamlit secrets. Please check your Streamlit Cloud secrets and Pinterest App settings.")
    st.stop()
if not PINTEREST_CLIENT_SECRET:
    st.error("❌ ERROR: PINTEREST_CLIENT_SECRET not found in Streamlit secrets. Please check your Streamlit Cloud secrets and Pinterest App settings.")
    st.stop()
if not REDIRECT_URI:
    st.error("❌ ERROR: REDIRECT_URI not found in Streamlit secrets. Please check your Streamlit Cloud secrets and Pinterest App settings (it must be your app's live URL).")
    st.stop()
if not GCP_PROJECT_ID:
    st.error("❌ ERROR: GCP_PROJECT_ID not found in Streamlit secrets. This is required for GCS operations.")
    st.stop()


# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("\n--- Gemini API configured successfully. ---")
except Exception as e:
    st.error(f"Error during initial Gemini API configuration: {e}. Please check your GEMINI_API_KEY.")
    print(f"\n--- ERROR: Initial Gemini API configuration failed: {e} ---")
    st.stop()

# Configure OpenAI API (for DALL-E 3)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
print("--- OpenAI API client initialized. ---")


# --- 2. Initialize Session State for Authentication ---
# This ensures that `st.session_state.pinterest_access_token` exists and is managed.
if 'pinterest_access_token' not in st.session_state:
    st.session_state.pinterest_access_token = None # Will be set after successful OAuth

if 'oauth_state' not in st.session_state:
    st.session_state.oauth_state = str(uuid.uuid4()) # Generate a unique state for CSRF protection


# --- 3. Handle OAuth Callback (This must be early in your script's execution flow) ---
# This logic executes when Pinterest redirects the user back to your app after authorization.
query_params = st.query_params

if "code" in query_params and "state" in query_params:
    auth_code = query_params["code"]
    received_state = query_params["state"]

    # --- CRITICAL SECURITY CHECK: Verify the 'state' parameter ---
    if received_state != st.session_state.oauth_state:
        st.error("❌ Security Error: State mismatch! Possible CSRF attack detected.")
        st.stop()

    st.info("Received authorization code from Pinterest. Exchanging for access token...")

    try:
        token_response = requests.post(
            PINTEREST_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": REDIRECT_URI,
            },
            auth=(PINTEREST_CLIENT_ID, PINTEREST_CLIENT_SECRET),
        )
        token_response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        token_data = token_response.json()
        st.session_state.pinterest_access_token = token_data.get("access_token")
        st.session_state.pinterest_refresh_token = token_data.get("refresh_token") # Store refresh token if provided
        st.session_state.pinterest_token_scope = token_data.get("scope") # Store scopes granted

        if st.session_state.pinterest_access_token:
            st.success("Pinterest authentication successful! Redirecting to main app...")
            # Clear query parameters and rerun to display the main app content
            st.query_params.clear() # Clear parameters from URL
            st.rerun() # Force a rerun
        else:
            st.error("Failed to obtain Pinterest access token from response. Please try again.")
            st.json(token_data) # Show response for debugging

    except requests.exceptions.RequestException as e:
        st.error(f"Error during token exchange: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Pinterest API Error: {e.response.status_code} - {e.response.text}")
    except json.JSONDecodeError:
        st.error("Failed to parse JSON response from Pinterest token endpoint.")

    st.stop() # Stop further execution while processing the callback and before rerun


# --- Helper Functions ---
def list_gemini_models():
    """Lists available Gemini models and their capabilities."""
    print("\n--- Listing Gemini models ---")
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"  Model: {m.name}")
    except Exception as e:
        print(f"--- ERROR: Could not list Gemini models: {e} ---")
        st.error(f"Could not list Gemini models: {e}. Check your GEMINI_API_KEY.")

def generate_image_with_dalle(prompt):
    """Generates an image using DALL-E 3."""
    print(f"\n--- Generating image with DALL-E 3 for prompt: '{prompt}' ---")
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        print(f"--- DALL-E 3 image generated: {image_url} ---")
        return image_url
    except Exception as e:
        print(f"--- ERROR: DALL-E 3 image generation failed: {e} ---")
        st.error(f"DALL-E 3 image generation failed: {e}. Please check your OPENAI_API_KEY or the prompt.")
        return None

def upload_to_gcs(bucket_name, source_image_url, destination_blob_name):
    """
    Uploads a file from a URL to a Google Cloud Storage bucket
    and returns its publicly accessible HTTP URL.
    """
    print(f"\n--- Uploading image from URL to GCS: {destination_blob_name} in {bucket_name} ---")
    try:
        # Use GCP_PROJECT_ID from st.secrets, not os.getenv
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        response = requests.get(source_image_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        blob.upload_from_string(response.content, content_type=response.headers['Content-Type'])

        # Construct the publicly accessible HTTP URL
        public_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
        print(f"--- Image uploaded to GCS and public URL generated: {public_url} ---")
        return public_url
    except Exception as e:
        print(f"--- ERROR: GCS upload failed: {e} ---")
        st.error(f"Failed to upload image to GCS: {e}. Check bucket name, GCS permissions, or source image URL.")
        return None

def save_dataframe_to_gcs(df, bucket_name, blob_name):
    """Saves a DataFrame to a JSONL file in Google Cloud Storage."""
    print(f"\n--- Saving DataFrame to GCS: {blob_name} in {bucket_name} ---")
    try:
        # Use GCP_PROJECT_ID from st.secrets, not os.getenv
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        jsonl_string = df.to_json(orient="records", lines=True)
        blob.upload_from_string(jsonl_string, content_type="application/jsonl")
        print(f"--- DataFrame saved successfully to GCS. ---")
        return True
    except Exception as e:
        print(f"--- ERROR: Failed to save DataFrame to GCS: {e} ---")
        st.error(f"Failed to save history to Google Cloud Storage: {e}. Data might not be persisted.")
        return False

def load_dataframe_from_gcs(bucket_name, blob_name):
    """Loads a DataFrame from a JSONL file in Google Cloud Storage."""
    print(f"\n--- Entering load_dataframe_from_gcs for bucket: {bucket_name}, blob: {blob_name} ---")
    try:
        # Use GCP_PROJECT_ID from st.secrets, not os.getenv
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            print(f"--- GCS Blob '{blob_name}' exists. Attempting to download... ---")
            downloaded_blob = blob.download_as_bytes()
            if downloaded_blob:
                try:
                    records = [json.loads(line) for line in downloaded_blob.decode('utf-8').splitlines() if line.strip()]
                    df = pd.DataFrame(records)
                    df = df.reindex(columns=TABLE_HEADERS, fill_value=None)
                    print(f"--- DataFrame loaded successfully from GCS. Shape: {df.shape} ---")
                    return df
                except json.JSONDecodeError as json_e:
                    print(f"--- ERROR: JSON decoding failed for {blob_name}: {json_e} ---")
                    st.error(f"Error decoding history file in GCS: {json_e}. Creating new history.")
                    return pd.DataFrame(columns=TABLE_HEADERS)
                except Exception as e:
                    print(f"--- ERROR: Unexpected error reading downloaded blob: {e} ---")
                    st.error(f"Error reading history from GCS: {e}. Creating new history.")
                    return pd.DataFrame(columns=TABLE_HEADERS)
            else:
                print(f"--- WARNING: Downloaded blob was empty for {blob_name}. ---")
                st.warning("History file in GCS is empty. Starting with fresh history.")
                return pd.DataFrame(columns=TABLE_HEADERS)
        else:
            print(f"--- GCS Blob '{blob_name}' does NOT exist. Returning empty DataFrame. ---")
            st.info("No existing history found in Google Cloud Storage. Starting fresh.")
            return pd.DataFrame(columns=TABLE_HEADERS)
    except Exception as e:
        print(f"--- FATAL ERROR in load_dataframe_from_gcs: {e} ---")
        st.error(f"Failed to load history from Google Cloud Storage: {e}. Please check your GCS bucket, blob name, and authentication (e.g., run 'gcloud auth application-default-login'). Starting with empty history.")
        return pd.DataFrame(columns=TABLE_HEADERS)


def get_pinterest_board_id(board_name, access_token):
    """Fetches the ID of a Pinterest board by its name."""
    url = "https://api-sandbox.pinterest.com/v5/boards"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"page_size": 100}
    print(f"\n--- Fetching Pinterest boards to find '{board_name}' ---")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        boards_data = response.json()
        for board in boards_data.get("items", []):
            if board.get("name") == board_name:
                print(f"--- Found Pinterest board '{board_name}' with ID: {board.get('id')} ---")
                return board.get("id")
        print(f"--- Pinterest board '{board_name}' not found. ---")
        return None
    except requests.exceptions.RequestException as e:
        print(f"--- ERROR: Pinterest API error fetching boards: {e} ---")
        st.error(f"Error fetching Pinterest boards: {e}. Check your Pinterest Access Token.")
        return None

def post_to_pinterest(image_url, title, description, board_id, access_token, link_url=""):
    """Posts an image and content to Pinterest."""
    print(f"\n--- Attempting to post to Pinterest board ID: {board_id} ---")
    if not access_token:
        st.error("Pinterest Access Token is missing. Cannot post to Pinterest.")
        return False, "Pinterest Access Token missing.", None
    if not board_id:
        st.error("Pinterest Board ID is missing. Cannot post to Pinterest.")
        return False, "Pinterest Board ID missing or not found.", None

    # Corrected Pinterest API URL for posting pins to SANDBOX
    url = "https://api-sandbox.pinterest.com/v5/pins"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    # Added 'name' field to the payload as per the 400 error message
    payload = {
        "board_id": board_id,
        "media_source": {"source_type": "image_url", "url": image_url},
        "title": title,
        "description": description,
        "name": title # Using title as a placeholder for 'name'
    }
    if link_url:
        payload["link"] = link_url
    try:
        print(f"--- Pinterest Pin Payload: {json.dumps(payload, indent=2)} ---") # Debug print
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        pin_data = response.json()
        pin_id = pin_data.get("id")
        pin_url = pin_data.get("pin_url")
        print(f"--- Pinterest Pin successful! Pin ID: {pin_id}, URL: {pin_url} ---")
        st.success(f"Pin posted successfully! [View Pin]({pin_url})")
        return True, pin_url, pin_id
    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error posting to Pinterest: {http_err.response.status_code} - {http_err.response.text}"
        print(f"--- ERROR: {error_message} ---")
        st.error(error_message)
        return False, None, None
    except requests.exceptions.RequestException as req_err:
        error_message = f"Network error posting to Pinterest: {req_err}"
        print(f"--- ERROR: {error_message} ---")
        st.error(error_message)
        return False, None, None
    except Exception as e:
        error_message = f"An unexpected error occurred while posting to Pinterest: {e}"
        print(f"--- ERROR: {error_message} ---")
        st.error(error_message)
        return False, None, None


def generate_seo_records_with_gemini(user_prompt):
    """
    Generates SEO-optimized Pinterest pin content using the Gemini model.
    Asks Gemini for a structured text output per record, then parses that text
    to construct valid Python dictionaries.
    """
    print(f"\n--- Entering generate_seo_records_with_gemini function for prompt: '{user_prompt}' ---")
    retries = 3
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            print(f"--- Attempting to use Gemini model: {model.model_name} (Attempt {attempt + 1}/{retries}) ---")

            prompt_template = f"""
            Generate 10 unique, SEO-optimized Pinterest pin content records for the topic: "{user_prompt}".

            **Objective:**
            - Content must be keyword-rich and highly relevant to the topic.
            - Ensure variety across the 10 records.
            - 'Image Background' must be highly descriptive for AI image generation.
            - Integrate primary and secondary keywords naturally.
            - Integrate 3-5 primary/secondary keywords in the description.
            - Hashtags should be relevant, trending, and niche.

            **Output Format (Strictly adhere to this block-based format for each record):**

            --- PIN RECORD 1 ---
            Title: [Your Title, Max 60 chars]
            Subtitle: [Your Subtitle, Max 120 chars]
            Hook: [Your Hook, Max 80 chars, Action-oriented]
            Image Background: [Highly detailed scene for AI image generation]
            Description: [Your Description, 200-500 chars, Integrate 3-5 primary/secondary keywords]
            Hashtags: #keyword1 #keyword2 #keyword3 #keyword4 #keyword5 #keyword6 #keyword7 #keyword8 #keyword9 #keyword10
            Alt Text: [Your Alt Text, Max 100 chars]

            --- PIN RECORD 2 ---
            Title: [Your Title, Max 60 chars]
            Subtitle: [Your Subtitle, Max 120 chars]
            Hook: [Your Hook, Max 80 chars, Action-oriented]
            Image Background: [Highly detailed scene for AI image generation]
            Description: [Your Description, 200-500 chars, Integrate 3-5 primary/secondary keywords]
            Hashtags: #keyword1 #keyword2 #keyword3 #keyword4 #keyword5 #keyword6 #keyword7 #keyword8 #keyword9 #keyword10
            Alt Text: [Your Alt Text, Max 100 chars]

            ... (Continue for 10 records)
            --- END RECORDS ---

            **Crucial:** Provide the **complete** output for all 10 records following this exact format. Do not truncate.
            Ensure 'Hashtags' are a single string of space-separated hashtags.
            """

            response = model.generate_content(prompt_template)
            raw_text_response = response.text.strip()
            print(f"--- Raw Gemini text response (FULL): \n{raw_text_response}\n ---")

            successful_records = []

            record_blocks = re.split(r'--- PIN RECORD \d+ ---', raw_text_response)
            record_blocks = [block.strip() for block in record_blocks if block.strip() and "--- END RECORDS ---" not in block]

            if not record_blocks:
                st.error("Gemini did not return any discernible pin records. Please try again.")
                return None

            print(f"--- Found {len(record_blocks)} potential record blocks. ---")

            for i, block in enumerate(record_blocks):
                current_record = {}
                lines = block.split('\n')

                print(f"--- Processing record block {i+1} (First 100 chars): {block[:100]}... ---")

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    match = re.match(r'^(.*?):\s*(.*)$', line)
                    if match:
                        key = match.group(1).strip()
                        value = match.group(2).strip()

                        if key == "Image Background":
                            current_record["Image Background"] = value
                        elif key == "Hashtags":
                            current_record["Hashtags"] = [h.strip() for h in value.split(' ') if h.strip()]
                        elif key in TABLE_HEADERS:
                            current_record[key] = value
                        else:
                            print(f"--- WARNING: Unknown key '{key}' found in record {i+1}. Skipping. ---")
                    else:
                        print(f"--- WARNING: Line could not be parsed in record {i+1}: '{line}'. Skipping line. ---")

                is_valid = True
                required_keys_in_output = ["Title", "Subtitle", "Hook", "Image Background", "Description", "Hashtags", "Alt Text"]
                for key in required_keys_in_output:
                    if key not in current_record or not current_record[key]:
                        print(f"--- WARNING: Record {i+1} is missing critical key '{key}' or value is empty. Skipping this record. ---")
                        st.warning(f"AI generation for record {i+1} was incomplete (missing '{key}'). This record will be skipped.")
                        is_valid = False
                        break

                if is_valid and not isinstance(current_record.get("Hashtags"), list):
                    print(f"--- WARNING: Record {i+1} has malformed 'Hashtags'. Expected list, got {type(current_record.get('Hashtags'))}. Skipping. ---")
                    st.warning(f"AI generation for record {i+1} has malformed 'Hashtags'. This record will be skipped.")
                    is_valid = False

                if is_valid:
                    successful_records.append(current_record)
                    print(f"--- Successfully parsed and validated record {i+1}. ---")
                else:
                    print(f"--- Record {i+1} was invalid and skipped. Current partial record: {current_record} ---")


            if not successful_records:
                st.error("No valid records could be parsed from Gemini's response. Please try again or refine your prompt.")
                return None

            print(f"--- Successfully collected {len(successful_records)} valid records after parsing. ---")
            return successful_records

        except Exception as e:
            print(f"--- AN OVERALL ERROR OCCURRED IN generate_seo_records_with_gemini (Attempt {attempt + 1}/{retries}): {e} ---")
            st.error(f"An overall error occurred during Gemini generation for attempt {attempt + 1}. Details in terminal.")
            if attempt < retries - 1:
                st.warning(f"Retrying Gemini generation due to overall error... (Attempt {attempt + 1}/{retries})")
                time.sleep(2)
                continue
            else:
                st.error(f"Error generating SEO records after {retries} attempts: {e}")
                st.error("Please check your Gemini API key, prompt quality, and model availability. See terminal for more details.")
                return None
    return None

# --- Main Streamlit Application Logic ---
def main():
    st.set_page_config(page_title="SEO & Pinterest Pin Generator", layout="wide")
    st.title("Automated SEO Content & Pinterest Pin Generator 🚀")
    st.markdown("Enter a topic, and let AI generate SEO-optimized content and Pinterest Pins for you!")

    with st.expander("Show Available Gemini Models (Debug Info)"):
        list_gemini_models()

    if 'content_generated' not in st.session_state:
        st.session_state.content_generated = False

    st.header("Previous Generations History")
    if 'history_df' not in st.session_state:
        st.session_state.history_df = load_dataframe_from_gcs(GCS_DATA_BUCKET, HISTORY_BLOB_NAME)
    if not st.session_state.history_df.empty:
        st.dataframe(st.session_state.history_df)
    else:
        st.info("No previous generations found. Generate some pins!")

    st.header("Generate New Pinterest Pins")
    user_topic = st.text_input("Enter your desired topic (e.g., 'natural hormone balance', 'eco-friendly cleaning tips'):", key="user_topic_input")

    generate_button = st.button("Generate Pins", key="generate_button")

    if generate_button and user_topic:
        st.subheader(f"Generating 10 pins for: '{user_topic}'")
        with st.spinner("Generating content with Gemini..."):
            generated_records = generate_seo_records_with_gemini(user_topic)

        if generated_records:
            st.success(f"Content generation complete! Successfully processed {len(generated_records)} records.")

            new_df = pd.DataFrame(generated_records)
            new_df["Image_Url"] = None
            new_df["Pinterest URL"] = None
            new_df["Status"] = "Pending"
            new_df["Board Name"] = "Not Selected"
            new_df["Pinterest Pin ID"] = None
            new_df = new_df.reindex(columns=TABLE_HEADERS, fill_value=None)

            if 'history_df' in st.session_state and not st.session_state.history_df.empty:
                    st.session_state.history_df = pd.concat([st.session_state.history_df, new_df], ignore_index=True)
            else:
                    st.session_state.history_df = new_df

            save_dataframe_to_gcs(st.session_state.history_df, GCS_DATA_BUCKET, HISTORY_BLOB_NAME)
            st.session_state.content_generated = True

    if st.session_state.content_generated and not st.session_state.history_df.empty:
        st.subheader("Generated Pins Preview:")
        st.dataframe(st.session_state.history_df)

        st.subheader("Generate Images & Post to Pinterest")
        st.info("Select a board name to proceed with image generation and Pinterest posting.")

        available_boards = []
        # Use st.session_state.pinterest_access_token here!
        if st.session_state.pinterest_access_token:
            boards_url = "https://api-sandbox.pinterest.com/v5/boards"
            headers = {"Authorization": f"Bearer {st.session_state.pinterest_access_token}"}
            try:
                boards_response = requests.get(boards_url, headers=headers)
                boards_response.raise_for_status()
                available_boards = [board.get("name") for board in boards_response.json().get("items", []) if board.get("name")]
                if not available_boards:
                    st.warning("No Pinterest boards found for the authenticated user. Please create a board on Pinterest first (using the API or Sandbox UI if available, but API is preferred for sandbox).")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not fetch Pinterest boards: {e}. Check your Pinterest Access Token.")
                available_boards = []
        else:
            st.warning("Pinterest Access Token not available. Please log in with Pinterest to fetch boards or post pins.")

        selected_board_name = st.selectbox("Select a Pinterest Board:", ["--- Select a Board ---"] + available_boards, key="board_select")

        if selected_board_name != "--- Select a Board ---":
            # Use st.session_state.pinterest_access_token here!
            board_id = get_pinterest_board_id(selected_board_name, st.session_state.pinterest_access_token)
            if board_id:
                st.success(f"Selected Board: '{selected_board_name}' (ID: {board_id})")
                pin_link_url = st.text_input(
                    "Optional: Enter a link URL for your pins (e.g., your blog post, product page):",
                    key="pin_link_url_input",
                    help="This URL will be added to all pins in this batch."
                )

                if st.button("Generate Images & Post All Pins to Selected Board", key="post_all_button"):
                    st.subheader("Processing Pins for Pinterest...")
                    pins_to_process_df = st.session_state.history_df[st.session_state.history_df["Status"] == "Pending"]

                    if pins_to_process_df.empty:
                        st.info("No pending pins to process. Generate new pins first or ensure previous pins are not already processed.")
                    else:
                        progress_bar = st.progress(0)
                        total_pins = len(pins_to_process_df)
                        for i, (idx, row) in enumerate(pins_to_process_df.iterrows()):
                            current_pin_display_number = i + 1
                            st.write(f"--- Processing Pin {current_pin_display_number}/{total_pins} ---")

                            if pd.isna(row["Image_Url"]) or not row["Image_Url"]:
                                image_prompt = row["Image Background"]
                                st.info(f"Generating image for Pin {current_pin_display_number} with DALL-E 3...")
                                dalle_image_url = generate_image_with_dalle(image_prompt)

                                if dalle_image_url:
                                    safe_title = re.sub(r'[^\w\s-]', '', row['Title']).strip().replace(' ', '_')[:50]
                                    blob_name = f"pinterest_pins/{safe_title}_{int(time.time())}.png"
                                    st.info(f"Uploading image for Pin {current_pin_display_number} to GCS...")
                                    gcs_image_url = upload_to_gcs(GCS_IMAGES_BUCKET, dalle_image_url, blob_name)
                                    if gcs_image_url:
                                        st.session_state.history_df.loc[idx, "Image_Url"] = gcs_image_url
                                        st.success(f"Image for Pin {current_pin_display_number} uploaded to GCS! ✅")
                                        st.image(dalle_image_url, caption=row["Alt Text"], width=200)
                                    else:
                                        st.error(f"Failed to upload image for Pin {current_pin_display_number} to GCS. Skipping Pinterest post for this pin. ❌")
                                        st.session_state.history_df.loc[idx, "Status"] = "Image Upload Failed"
                                        continue
                                else:
                                    st.error(f"Failed to generate image for Pin {current_pin_display_number}. Skipping Pinterest post for this pin. ❌")
                                    st.session_state.history_df.loc[idx, "Status"] = "Image Generation Failed"
                                    continue
                            else:
                                gcs_image_url = row["Image_Url"]
                                st.info(f"Using existing image for Pin {current_pin_display_number}: {gcs_image_url}")
                                st.image(gcs_image_url, caption=row["Alt Text"], width=200)

                            if gcs_image_url:
                                st.info(f"Posting Pin {current_pin_display_number} to Pinterest...")
                                posted, pinterest_url, pin_id = post_to_pinterest(
                                    image_url=gcs_image_url,
                                    title=row["Title"],
                                    description=row["Description"],
                                    board_id=board_id,
                                    access_token=st.session_state.pinterest_access_token, # Use the token from session_state
                                    link_url=pin_link_url
                                )
                                if posted:
                                    st.session_state.history_df.loc[idx, "Pinterest URL"] = pinterest_url
                                    st.session_state.history_df.loc[idx, "Status"] = "Posted"
                                    st.session_state.history_df.loc[idx, "Board Name"] = selected_board_name
                                    st.session_state.history_df.loc[idx, "Pinterest Pin ID"] = pin_id
                                    st.success(f"Pin {current_pin_display_number} posted successfully! 🎉")
                                else:
                                    st.session_state.history_df.loc[idx, "Status"] = "Pinterest Post Failed"
                                    st.error(f"Failed to post Pin {current_pin_display_number} to Pinterest. ❌")
                            else:
                                st.error(f"No image URL available for Pin {current_pin_display_number}. Cannot post to Pinterest. 🛑")
                                st.session_state.history_df.loc[idx, "Status"] = "No Image for Post"

                            progress_bar.progress((current_pin_display_number) / total_pins)
                            save_dataframe_to_gcs(st.session_state.history_df, GCS_DATA_BUCKET, HISTORY_BLOB_NAME)

                        st.success("All selected pins processed! ✨")
                        st.dataframe(st.session_state.history_df)

    # Optional: Logout button
    if st.session_state.pinterest_access_token: # Only show logout if authenticated
        if st.sidebar.button("Logout from Pinterest"):
            del st.session_state.pinterest_access_token
            if 'pinterest_refresh_token' in st.session_state:
                del st.session_state.pinterest_refresh_token
            if 'oauth_state' in st.session_state: # Clear the state on logout
                del st.session_state.oauth_state
            st.success("Logged out from Pinterest.")
            st.rerun() # Rerun to go back to the login state


if __name__ == "__main__":
    main()