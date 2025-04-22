✨ Tired of Invisible Pixel Pests & Sneaky AI Trackers? Introducing the Pixel Woodchipper Pro™! ✨

Are your legitimately created images being held hostage by invisible digital meddlers? Is your AI training data potentially "poisoned" by techniques like Glaze™ or Nightshade™? Or perhaps worse, are images you paid for, or generated yourself, secretly embedded with invisible watermarks by AI platforms trying to track their output without your knowledge?

Fight back against digital shenanigans! Unleash the Pixel Woodchipper Pro™, our state-of-the-art* (state-of-the-art circa 1998 GIF compression) Python purification ritual!

Watch in amazement as we take your image and treat it with the digital equivalent of feeding it through an industrial shredder! Yes, we lovingly disassemble it, forcing those pixels through rigorous quantization gauntlets like they're applying for a low-res GIF visa... multiple times! We're not just sanding off the Glaze™; we're aiming to scramble those hidden AI tracking codes too!

But wait, there's more! We then blast it with scaling shenanigans, stretching and squishing it like digital taffy, applying just a touch of blur here, a dash of sharpening there – think of it as aggressively power-washing away those unwanted fingerprints!

Finally, our patented* (patent definitely not pending) High/Low Pass Filter-Blender-Tron 5000™ meticulously* (results may vary, wildly) slaps the remaining pixel dust back together with the digital equivalent of duct tape and hope!

The result? An image stunningly similar to the original (if you squint, after a long day). But most importantly, it's aiming to be SANITIZED! CLEANSED! LIBERATED! from unwanted tracking watermarks and disruptive data perturbations!

Disclaimer: The Pixel Woodchipper Pro™ views "non-destructive" as a mere suggestion. Its primary goal is MAXIMUM DISRUPTION to hidden data structures. While we aim for something visually passable, consider this a full digital exorcism, not a gentle spa day for your pixels.

Get the Pixel Woodchipper Pro™ today! Free your images through glorious, intentional algorithmic chaos! (Requires Python. Sense of humour highly recommended.)

(Seriously though...)

This project is provided completely open-source under the MIT License. Feel free to use, modify, and distribute it according to the license terms.

Technical Description of the Process:

This Python script applies a sequence of aggressive image manipulation techniques. Its primary goal is to potentially disrupt embedded, invisible watermarks (such as those theorized or known to be used by some AI image generation platforms for output tracking) or data perturbations intended to interfere with AI model training (like Glaze or Nightshade). It achieves this by significantly altering the image's pixel data using a multi-stage approach, while attempting to retain a visually recognizable result.

The process involves:

Multi-Palette Quantization & Scaling: Aggressively reduces color depth using multiple algorithms (Median Cut, Max Coverage, Fast Octree) and dithering methods (Ordered, Floyd-Steinberg) at the original, 2x, and 3x scales. These scaled versions are then slightly blurred or sharpened before being downscaled and averaged, introducing significant statistical changes to pixel values and relationships.

Filtering and Detail Reconstruction: Applies histogram equalization and high/low-pass filtering combined with blurring and detail re-addition to further manipulate the pixel data, particularly targeting frequency domain characteristics where some watermarks might reside.

Blending: Combines the results of the quantization/scaling and filtering stages.

Optional Denoising: Applies a Median filter blended back into the image, targeting small, high-frequency noise patterns sometimes used in steganography or potentially smoothing artifacts from previous stages.

Debug Output: Allows inspection of intermediate steps.

Important Note on Efficacy: The combination of techniques employed is designed to be highly disruptive to a wide range of potential data hiding methods by attacking the image data through color reduction, spatial manipulation (scaling, blur, sharpen), dithering noise, and frequency domain adjustments. However, there is absolutely no guarantee that this process will successfully remove or disrupt all conceivable forms of watermarking or data perturbation, especially sophisticated or unknown future methods. Watermarking is a complex field, and techniques resistant to various transformations exist. This script represents a forceful attempt at disruption based on common image processing operations, significantly increasing the likelihood of impacting many embedded data techniques, but it is not a guaranteed universal solution.

Run:



    python watermark_disruptor.py path/to/your/input_image.png path/to/your/output_image.png
    
    No denoising (default):
    python watermark_disruptor_v5_denoise.py input.png output.png

    Apply denoising blend with 30% strength:
    python watermark_disruptor_v5_denoise.py input.png output.png --denoise 0.3

    Apply full denoising (output is just the median filtered image):
    python watermark_disruptor_v5_denoise.py input.png output.png --denoise 1.0

    With debug and denoising:
    python watermark_disruptor_v5_denoise.py input.png output.png --denoise 0.5 --debug
