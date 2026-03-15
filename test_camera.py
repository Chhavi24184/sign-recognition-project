import cv2

def test_cameras():
    print("Testing for available cameras with different backends...")
    available_cameras = []
    
    backends = [
        ("Default", cv2.CAP_ANY),
        ("MSMF (Win 10+)", cv2.CAP_MSMF),
        ("DirectShow", cv2.CAP_DSHOW)
    ]
    
    for backend_name, backend_id in backends:
        print(f"\n--- Testing Backend: {backend_name} ---")
        for i in range(3):
            cap = cv2.VideoCapture(i + backend_id)
            if cap.isOpened():
                print(f"Index {i}: [OK]")
                available_cameras.append((i, backend_name))
                cap.release()
            else:
                print(f"Index {i}: [NOT FOUND]")
            
    if available_cameras:
        print(f"\nSuccessfully found cameras at indices: {available_cameras}")
    else:
        print("\nNo cameras found! Please check your hardware connection.")

if __name__ == "__main__":
    test_cameras()
