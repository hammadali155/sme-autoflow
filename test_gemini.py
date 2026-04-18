import os
import sys

try:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import errors
except ImportError as e:
    print(f"[X] Missing required library: {e}")
    print("Please make sure your virtual environment is activated and dependencies are installed.")
    sys.exit(1)

def test_gemini_connection():
    print("[*] Testing Gemini API Connection...")
    
    # Load environment variables from .env file
    if not load_dotenv(".env"):
        print("[!] Warning: Could not open .env file. Checking system environment variables instead.")
    else:
        print("[v] Loaded .env file.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("[X] Error: GEMINI_API_KEY is not set in your .env file or environment.")
        sys.exit(1)

    print(f"[*] Found GEMINI_API_KEY: {api_key[:6]}...{api_key[-4:]}")

    try:
        # Initialize the client
        client = genai.Client(api_key=api_key)
        
        # We use gemini-2.0-flash as this is what your project relies on
        model_name = "gemini-2.5-flash"
        print(f"[*] Sending a test request to {model_name}...")
        
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'Hello from Gemini!' and nothing else."
        )
        
        print("[v] Success! Gemini is responding.")
        print("-" * 40)
        print("Gemini says:", response.text)
        print("-" * 40)
        print("Your API key is working perfectly and has sufficient quota.")

    except errors.APIError as e:
        status_code = getattr(e, 'code', 'Unknown Code')
        print(f"\n[X] API Error occurred (Code: {status_code})")
        print(f"Error Details: {e.message if hasattr(e, 'message') else str(e)}")
        
        if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower() or status_code == 429:
            print("\n[!] Diagnosis: QUOTA EXCEEDED")
            print("Your API key is valid, but it has reached its free tier limit.")
            print("To fix this, go to Google AI Studio and attach a billing account to your project.")
        elif "INVALID_ARGUMENT" in str(e) or "API_KEY_INVALID" in str(e).lower() or status_code == 400:
            print("\n[!] Diagnosis: INVALID API KEY")
            print("The API key provided is not valid, missing, or expired.")
        else:
            print("\n[!] Diagnosis: UNKNOWN ERROR")
            print("Please check the error details above.")

        # # Iterate through the models
        # for model in client.models.list():
        #     print(f"ID: {model.name}")
        #     print(f"Name: {model.display_name}")
        #     print(f"Description: {model.description}")
        #     print("-" * 30)
    except Exception as e:
        print(f"\n[X] An unexpected error occurred: {type(e).__name__} - {str(e)}")

    print("\n")

if __name__ == "__main__":
    test_gemini_connection()
