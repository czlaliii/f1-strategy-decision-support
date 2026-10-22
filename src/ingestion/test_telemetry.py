from src.ingestion.api_client import OpenF1Client

SESSION_KEY = 11342
DRIVER_NUMBER = 1

def main():

    client = OpenF1Client()

    print("Testing car_data...")

    car_data = client.get(
        "car_data",
        {
            "session_key": SESSION_KEY,
            "driver_number": DRIVER_NUMBER
        }
    )

    print(f"Car data records: {len(car_data)}")

    print("\nFirst record:")
    print(car_data[0])

    print("\nTesting location...")

    location = client.get(
        "location",
        {
            "session_key": SESSION_KEY,
            "driver_number": DRIVER_NUMBER
        }
    )

    print(f"Location records: {len(location)}")

    print("\nFirst record:")
    print(location[0])


if __name__ == "__main__":
    main()