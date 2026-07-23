# Robot Vision: Adaptive Visual Grounding

This project developed as a natural offshoot of building robot-vision capabilities. A robot must first connect a human instruction to the correct object in its visual field before it can plan an interaction. This system explores that grounding step by turning natural-language requests into localized image targets that can also be followed through recorded or live video.

The interactive application compares local and GPT-guided visual grounding while keeping still-image attention as the primary research focus.

## Capabilities

- Natural-language target selection in uploaded images
- YOLO detection for supported object classes
- OWL-ViT open-vocabulary grounding
- GPT-guided OWL-ViT for attributes, relationships, and ambiguous scenes
- Quantity and spatial-relation parsing
- Optional box-prompted segmentation
- Single-target tracking in recorded video
- Browser-camera tracking through WebRTC
- Downloadable annotated results and structured prediction data

## System overview

An instruction is parsed into a target, attributes, quantity, and spatial relationships. The router then selects an appropriate backend:

- **YOLO** provides faster detection for known object classes.
- **OWL-ViT** supports open-vocabulary image grounding.
- **GPT-guided OWL-ViT** uses local candidates and a single guided vision request when a scene requires attribute or relationship reasoning.

Model files are not stored in GitHub. Each backend downloads its pinned model from the original provider when first selected and remains cached for the running application process. The first use of a backend may therefore take several minutes.

## GPT-guided mode

GPT-guided OWL-ViT can use either a deployment-level `OPENAI_API_KEY` or a key entered in the Streamlit sidebar. A user-entered key is scoped to that Streamlit session and is not written to files or request logs.

When GPT-guided mode is selected, the relevant candidate image is sent to the OpenAI API for visual selection. YOLO and OWL-ViT remain fully local after their model files have been downloaded.

## Limitations

- Still-image grounding is the primary focus. Recorded-video and live-camera bounding boxes can be less precise because detections are combined with tracking between inference cycles.
- The camera should remain stable during initial grounding. Initial inference can take several seconds on CPU or shared cloud compute because of limited compute availability and model cost.
- Clear camera quality, appropriate lighting, limited motion blur, and an unobstructed target improve results.
- Segmentation is experimental. It significantly reduces recorded-video and live-camera performance and is not recommended for video demonstrations.
- The system is a research prototype and is not intended for safety-critical robot control.

## Future work

- Dedicated cloud inference backend
- ROS or ROS 2 deployment package
- Faster live inference through GPU execution, quantization, and asynchronous processing
- Improved video-box stability and target reacquisition
- Real-time segmentation improvements
- Evaluation across broader lighting, camera, and scene conditions

## Run locally

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Models download automatically when first selected. An OpenAI API key is required only for GPT-guided OWL-ViT and can be entered directly in the application.

## Validation

The repository includes automated tests for routing, parsing, bounding-box safeguards, API behavior, video target selection, and Streamlit result rendering. GitHub Actions runs the suite on supported Python versions.

## License

Project source is available under the [MIT License](LICENSE). Model files and third-party services retain their own licenses and usage terms.
