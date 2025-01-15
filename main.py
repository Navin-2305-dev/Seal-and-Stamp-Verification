import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_image_for_resnet(image, size=(224, 224)):
    try:
        resized = cv2.resize(image, size)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        normalized_image = tf.keras.applications.resnet50.preprocess_input(rgb_image)
        return np.expand_dims(normalized_image, axis=0) 
    except Exception as e:
        raise ValueError(f"Error in image preprocessing for ResNet: {e}")

def extract_features(image, model):
    try:
        features = model.predict(image)
        return features
    except Exception as e:
        raise ValueError(f"Error in feature extraction: {e}")

def load_resnet_model():
    base_model = tf.keras.applications.ResNet101(weights="imagenet", include_top=False, pooling="avg")
    return base_model

def compute_cosine_similarity(features1, features2):
    try:
        similarity = cosine_similarity(features1, features2)
        return similarity[0][0] * 100
    except Exception as e:
        raise ValueError(f"Error in similarity computation: {e}")

def preprocess_image_for_cv(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(morph)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(mask, [contour], -1, 255, -1)

        return mask
    except Exception as e:
        raise ValueError(f"Error in preprocessing for traditional CV: {e}")

def detect_and_match_features(img1, img2):
    try:
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 10:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
        else:
            matches_mask = []

        similarity_score = len(matches_mask) / len(good_matches) * 100 if good_matches else 0

        match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                    matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return similarity_score, match_img, len(good_matches), len(matches_mask)

    except Exception as e:
        raise ValueError(f"Error in feature matching: {e}")

def main():
    st.title("Advanced Signature Similarity Checker")
    st.write("Upload two signature images to detect and compare their similarity using both traditional and deep learning techniques.")

    uploaded_file1 = st.file_uploader("Upload the first signature image", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload the second signature image", type=["jpg", "jpeg", "png"])

    if uploaded_file1 and uploaded_file2:
        try:
            model = load_resnet_model()

            img1 = Image.open(uploaded_file1)
            img2 = Image.open(uploaded_file2)
            
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

            st.image([img1, img2], caption=["First Image", "Second Image"], width=300)

            img1_preprocessed_cv = preprocess_image_for_cv(img1_cv)
            img2_preprocessed_cv = preprocess_image_for_cv(img2_cv)

            similarity_cv, match_img, good_matches, total_matches = detect_and_match_features(img1_preprocessed_cv, img2_preprocessed_cv)

            st.image(match_img, caption="Feature Matches (Traditional CV)", channels="BGR", width=600)
            st.write(f"Good Matches: {good_matches}/{total_matches}")
            
            img1_preprocessed_dl = preprocess_image_for_resnet(img1_cv)
            img2_preprocessed_dl = preprocess_image_for_resnet(img2_cv)

            features1 = extract_features(img1_preprocessed_dl, model)
            features2 = extract_features(img2_preprocessed_dl, model)

            combined_score = compute_cosine_similarity(features1, features2)

            st.write(f"Similarity Score: {combined_score:.2f}%")

            if combined_score > 80:
                st.success("Signatures are highly similar. Likely authentic.")
            elif combined_score > 50:
                st.warning("Signatures are moderately similar. Verify further.")
            else:
                st.error("Signatures are dissimilar. Likely forged.")

        except UnidentifiedImageError:
            st.error("One or both uploaded files are not valid images. Please upload valid image files.")
        except ValueError as ve:
            st.error(f"An error occurred: {ve}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
