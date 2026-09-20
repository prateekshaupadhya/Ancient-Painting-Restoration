import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import UNet
import io


# Page settings
st.set_page_config(
    page_title="Ancient Painting Restoration",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Ancient Painting Restoration")
st.write("Upload a damaged painting and its damage mask.")


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
model = UNet()

model_path = "models/best_masked_unet_v2.pth"

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.to(device)
model.eval()


# Upload damaged painting
damaged_file = st.file_uploader(
    "Upload Damaged Painting",
    type=["jpg", "jpeg", "png"]
)

# Upload mask
mask_file = st.file_uploader(
    "Upload Damage Mask",
    type=["jpg", "jpeg", "png"]
)


if damaged_file and mask_file:

    # Read images
    damaged_image = Image.open(damaged_file).convert("RGB")
    mask_image = Image.open(mask_file).convert("L")

    st.subheader("Input Images")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            damaged_image,
            caption="Damaged Painting",
            use_container_width=True
        )

    with col2:
        st.image(
            mask_image,
            caption="Damage Mask",
            use_container_width=True
        )


    # Remember original size
    original_size = damaged_image.size


    # Resize for model
    damaged_resized = damaged_image.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    mask_resized = mask_image.resize(
        (256, 256),
        Image.Resampling.NEAREST
    )


    # Convert damaged image
    damaged_array = (
        np.array(damaged_resized) / 255.0
    )


    # Convert mask
    mask_array = (
        np.array(mask_resized) / 255.0
    )

    # Binary mask
    mask_array = (
        mask_array > 0.5
    ).astype(np.float32)


    # Convert to tensors
    damaged_tensor = torch.tensor(
        damaged_array,
        dtype=torch.float32
    ).permute(2, 0, 1)

    mask_tensor = torch.tensor(
        mask_array,
        dtype=torch.float32
    ).unsqueeze(0)


    # 4-channel input:
    # RGB + mask
    input_tensor = torch.cat(
        [damaged_tensor, mask_tensor],
        dim=0
    ).unsqueeze(0)


    input_tensor = input_tensor.to(device)


    # Model prediction
    with torch.no_grad():
        prediction = model(input_tensor)


    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )


    # Convert prediction to image
    prediction_image = Image.fromarray(
        (prediction * 255).astype(np.uint8)
    )


    # Resize back to original size
    prediction_image = prediction_image.resize(
        original_size,
        Image.Resampling.BILINEAR
    )


    prediction_array = (
        np.array(prediction_image) / 255.0
    )


    # Original damaged image
    damaged_original = (
        np.array(damaged_image) / 255.0
    )


    # Resize/use original mask
    mask_original = (
        np.array(mask_image) / 255.0
    )

    mask_original = (
        mask_original > 0.5
    ).astype(np.float32)

    mask_original = mask_original[..., np.newaxis]


    # IMPORTANT:
    # Restore ONLY the damaged region.
    # Keep the undamaged region unchanged.
    final_image = (
        prediction_array * mask_original
        +
        damaged_original * (1 - mask_original)
    )


    final_image = np.clip(
        final_image * 255,
        0,
        255
    ).astype(np.uint8)


    # Display result
    st.subheader("✨ Restored Painting")

    st.image(
        final_image,
        caption="AI Restored Painting",
        use_container_width=True
    )


    # Download button
    result_image = Image.fromarray(final_image)

    buffer = io.BytesIO()

    result_image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Restored Painting",
        data=buffer.getvalue(),
        file_name="restored_painting.png",
        mime="image/png"
    )