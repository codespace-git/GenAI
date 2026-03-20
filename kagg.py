import os
import pandas as pd
import cv2
from glasses_detector import GlassesClassifier

clf = GlassesClassifier()

root_path = "kagglehub/datasets/jeffheaton/glasses-or-no-glasses/versions/2/"
df = pd.read_csv(root_path + "train.csv")

corrected_images = {
    "image-no": [],
    "final-label": [],
}

prefix = "face-"
labels = ["no-glasses", "glasses"]
images = root_path + "faces-spring-2020/faces-spring-2020"

for idx, row in df.iterrows():
    label = int(row["glasses"])
    img_name = prefix + f"{idx + 1}.png"
    img_path = os.path.join(images, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    predicted = clf.predict(image=img_path, format="int")

    if label != predicted:
        print(f"\nMismatch at {idx+1}")
        print(f"Predicted: {predicted}, Actual: {label}")

        display = img.copy()
        cv2.putText(display, f"Label: {labels[label]}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Pred: {labels[predicted]}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Mismatch", display)

        key = cv2.waitKey(0)

        if key == ord('q'):
            break

        elif key == ord('m'):  # flip label
            new_label = 1 - label
            df.at[idx, "glasses"] = new_label

            corrected_images["image-no"].append(idx + 1)
            corrected_images["final-label"].append(new_label)

            print("✔ Corrected")

        elif key == ord('s'):
            print("Skipped")

        cv2.destroyAllWindows()

    if idx % 100 == 0:
        df.to_csv("train-corrected.csv", index=False)
        print("💾 Autosaved...")

# Final save
df.to_csv("train-corrected.csv", index=False)
print("✅ Done")

cdf = pd.DataFrame(corrected_images)
cdf.to_csv("correction-log.csv", index=False)