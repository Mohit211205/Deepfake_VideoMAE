import gradio as gr
from src.inference import predict_deepfake

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Video(label="Upload Video to Test"),
    outputs=[
        gr.Label(label="True VideoMAE Prediction"),
        gr.Image(label="Explainability (Attention Heatmap)"),
        gr.Image(label="Frame-by-Frame Suspicion Graph")
    ],
    title="Domain-Invariant Deepfake Detection 🛡️",
    description="""
    Upload any face video to detect if it is **Real** or a **Deepfake**.
    
    This model is powered by a **True VideoMAE Backbone** trained with **Multi-View Consistency (Domain SSL)**, 
    making it robust to compression artifacts and unseen domains.
    
    *The output generates an attention heatmap and a frame-by-frame graph showing exactly when the model detected fake artifacts.*
    """
)

if __name__ == "__main__":
    # Launch locally
    print("Launching Gradio App Locally...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
