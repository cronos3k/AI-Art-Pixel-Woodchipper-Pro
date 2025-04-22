import sys
import argparse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
from scipy.ndimage import gaussian_filter
import os

# --- Debug Saving Helper ---
def save_debug_image(img_to_save, debug_enabled, step_counter_list, debug_dir, description):
    """Saves an image to the debug directory if debug is enabled."""
    if not debug_enabled or not img_to_save:
        return

    try:
        # Ensure it's a PIL Image object
        if isinstance(img_to_save, np.ndarray):
            try:
                if img_to_save.ndim == 3 and img_to_save.shape[2] in [3, 4]:
                     img_pil = Image.fromarray(img_to_save.astype(np.uint8))
                elif img_to_save.ndim == 2:
                     img_pil = Image.fromarray(img_to_save.astype(np.uint8), 'L')
                else:
                    print(f"  Debug save: Skipping NumPy array with unexpected shape: {description}")
                    return
            except Exception as conversion_e:
                 print(f"  Debug save: Failed to convert NumPy array to PIL Image: {description}. Error: {conversion_e}")
                 return
        elif isinstance(img_to_save, Image.Image):
            img_pil = img_to_save
        else:
             print(f"  Debug save: Skipping non-image/non-numpy type ({type(img_to_save)}): {description}")
             return

        # Increment counter
        step_counter_list[0] += 1
        step_str = f"{step_counter_list[0]:03d}"

        # Sanitize description
        safe_desc = "".join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in description)
        safe_desc = safe_desc[:50]
        filename = f"{step_str}_{safe_desc}.png"
        filepath = os.path.join(debug_dir, filename)

        # Handle modes like 'P'
        saveable_img = img_pil
        if saveable_img.mode == 'P':
            saveable_img = saveable_img.convert('RGB')
        # Ensure RGBA is handled correctly by save
        if saveable_img.mode == 'RGBA':
             # PNG supports RGBA directly
             pass
        elif saveable_img.mode != 'RGB' and saveable_img.mode != 'L':
             print(f"  Debug save: Converting mode {saveable_img.mode} to RGB for saving {filename}")
             saveable_img = saveable_img.convert('RGB')


        print(f"  Saving debug image: {filepath}")
        saveable_img.save(filepath, 'PNG')

    except Exception as e:
        print(f"  Warning: Failed to save debug image '{description}'. Error: {e}")


# --- Core Helper Functions (Quantize, Filters) ---

def quantize_image(img, colors=256, method=Image.Quantize.MEDIANCUT, dither_method=Image.Dither.FLOYDSTEINBERG):
    quantized = img.quantize(colors=colors, method=method, kmeans=0, dither=dither_method)
    return quantized.convert('RGB')

def high_pass_filter(img_array, sigma):
    low_pass = gaussian_filter(img_array, sigma=(sigma, sigma, 0), mode='nearest', truncate=4.0)
    low_pass = low_pass.astype(img_array.dtype)
    high_pass = img_array - low_pass
    return high_pass

def low_pass_filter(img_array, sigma):
     low_pass = gaussian_filter(img_array, sigma=(sigma, sigma, 0), mode='nearest', truncate=4.0)
     return low_pass.astype(np.uint8)


# --- Multi-Palette Quantization (Includes Debug) ---

def apply_multi_palette_quantization_blend(img, colors=256, debug_enabled=False, step_counter_list=None, debug_dir=None, base_desc=""):
    palette_methods = {
        "MEDIANCUT": Image.Quantize.MEDIANCUT,
        "MAXCOVERAGE": Image.Quantize.MAXCOVERAGE,
        "FASTOCTREE": Image.Quantize.FASTOCTREE
    }
    palette_results_arrays = []
    base_img_rgb = img.convert('RGB')

    print(f"    Starting multi-palette quantization ({colors} colors) for: {base_desc}")
    for name, method_enum in palette_methods.items():
        # print(f"      Processing with palette: {name}") # Reduced verbosity
        try:
            quantized_pattern = quantize_image(base_img_rgb, colors=colors, method=method_enum, dither_method=Image.Dither.ORDERED)
            save_debug_image(quantized_pattern, debug_enabled, step_counter_list, debug_dir, f"{base_desc}_quant_{name}_pattern")

            quantized_noise = quantize_image(base_img_rgb, colors=colors, method=method_enum, dither_method=Image.Dither.FLOYDSTEINBERG)
            save_debug_image(quantized_noise, debug_enabled, step_counter_list, debug_dir, f"{base_desc}_quant_{name}_noise")

            blended_dither_result = Image.blend(quantized_pattern, quantized_noise, 0.5)
            save_debug_image(blended_dither_result, debug_enabled, step_counter_list, debug_dir, f"{base_desc}_quant_{name}_blended")

            palette_results_arrays.append(np.array(blended_dither_result).astype(np.float32))
        except Exception as e:
            print(f"      Warning: Failed to quantize with {name}. Skipping. Error: {e}")
            continue

    if not palette_results_arrays:
        print("      Error: No successful quantization results to blend. Returning original.")
        save_debug_image(base_img_rgb, debug_enabled, step_counter_list, debug_dir, f"{base_desc}_quant_FAILED_original")
        return base_img_rgb

    # print(f"    Blending results from different palettes for: {base_desc}") # Reduced verbosity
    average_array = np.mean(np.array(palette_results_arrays), axis=0)
    final_blended_array = np.clip(average_array, 0, 255).astype(np.uint8)
    final_blended_image = Image.fromarray(final_blended_array, 'RGB')

    save_debug_image(final_blended_image, debug_enabled, step_counter_list, debug_dir, f"{base_desc}_quant_all_palettes_blended")
    # print(f"    Multi-palette quantization blending finished for: {base_desc}") # Reduced verbosity
    return final_blended_image


# --- Scaled Quantization Orchestrator (Includes Debug) ---

def apply_scaled_quantization_and_blend(img_orig, blur_sigma_downscale=0.5, sharpen_radius=1, sharpen_percent=150, sharpen_threshold=3, debug_enabled=False, step_counter_list=None, debug_dir=None):
    original_size = img_orig.size
    w, h = original_size
    img_rgb = img_orig.convert('RGB')
    results_to_blend = []

    # 1. Process at 1x scale
    print("  Processing at 1x scale...")
    result_1x = apply_multi_palette_quantization_blend(img_rgb, colors=256, debug_enabled=debug_enabled, step_counter_list=step_counter_list, debug_dir=debug_dir, base_desc="scale1x")
    results_to_blend.append(np.array(result_1x).astype(np.float32))
    print("  1x scale processing done.")

    # 2. Process at 2x scale
    print("  Processing at 2x scale...")
    upscaled_dims_2x = (w * 2, h * 2)
    img_upscaled_2x = img_rgb.resize(upscaled_dims_2x, Image.Resampling.LANCZOS)
    save_debug_image(img_upscaled_2x, debug_enabled, step_counter_list, debug_dir, "scale2x_upscaled")
    quant_2x = apply_multi_palette_quantization_blend(img_upscaled_2x, colors=256, debug_enabled=debug_enabled, step_counter_list=step_counter_list, debug_dir=debug_dir, base_desc="scale2x")
    quant_2x_blurred = quant_2x.filter(ImageFilter.GaussianBlur(radius=blur_sigma_downscale))
    save_debug_image(quant_2x_blurred, debug_enabled, step_counter_list, debug_dir, "scale2x_quant_blurred")
    result_2x_blurred_down = quant_2x_blurred.resize(original_size, Image.Resampling.LANCZOS)
    save_debug_image(result_2x_blurred_down, debug_enabled, step_counter_list, debug_dir, "scale2x_quant_blurred_downscaled")
    results_to_blend.append(np.array(result_2x_blurred_down).astype(np.float32))
    print("  2x scale processing done.")

    # 3. Process at 3x scale
    print("  Processing at 3x scale...")
    upscaled_dims_3x = (w * 3, h * 3)
    img_upscaled_3x = img_rgb.resize(upscaled_dims_3x, Image.Resampling.LANCZOS)
    save_debug_image(img_upscaled_3x, debug_enabled, step_counter_list, debug_dir, "scale3x_upscaled")
    quant_3x = apply_multi_palette_quantization_blend(img_upscaled_3x, colors=256, debug_enabled=debug_enabled, step_counter_list=step_counter_list, debug_dir=debug_dir, base_desc="scale3x")
    quant_3x_sharpened = quant_3x.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=sharpen_threshold))
    save_debug_image(quant_3x_sharpened, debug_enabled, step_counter_list, debug_dir, "scale3x_quant_sharpened")
    result_3x_sharpened_down = quant_3x_sharpened.resize(original_size, Image.Resampling.LANCZOS)
    save_debug_image(result_3x_sharpened_down, debug_enabled, step_counter_list, debug_dir, "scale3x_quant_sharpened_downscaled")
    results_to_blend.append(np.array(result_3x_sharpened_down).astype(np.float32))
    print("  3x scale processing done.")

    # 4. Blend the three scale results equally
    print("  Blending 1x, downscaled 2x, and downscaled 3x results...")
    if len(results_to_blend) < 1:
        raise ValueError("No results generated from scaled processing for blending.")
    if len(results_to_blend) < 3:
         print(f"  Warning: Only {len(results_to_blend)} scale results available for blending.")

    average_array = np.mean(np.array(results_to_blend), axis=0)
    final_blended_array = np.clip(average_array, 0, 255).astype(np.uint8)
    final_scaled_quant_blended_image = Image.fromarray(final_blended_array, 'RGB')
    save_debug_image(final_scaled_quant_blended_image, debug_enabled, step_counter_list, debug_dir, "STAGE1_RESULT_scaled_quant_blended")
    print("  Scaled quantization and blending stage finished.")
    return final_scaled_quant_blended_image


# --- Main Processing Function (Modified for Denoise) ---

def process_image(input_path, output_path, filter_blend_alpha=0.5, filter_sigma=1.5, debug_enabled=False, denoise_strength=0.0):
    """
    Applies the V5 process (Multi-Palette Quant/Scale/Filter + Denoise + Debug) to disrupt watermarks.
    """
    step_counter_list = [0]
    debug_dir = None

    if debug_enabled:
        output_dir = os.path.dirname(output_path)
        output_filename_base = os.path.splitext(os.path.basename(output_path))[0]
        debug_dir = os.path.join(output_dir if output_dir else '.', f"{output_filename_base}_debug_steps")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode enabled. Saving intermediate steps to: {debug_dir}")
        except OSError as e:
            print(f"Warning: Could not create debug directory '{debug_dir}'. Disabling debug output. Error: {e}")
            debug_enabled = False

    try:
        print(f"Loading image: {input_path}")
        img_orig_pil = Image.open(input_path)
        original_mode = img_orig_pil.mode
        original_size = img_orig_pil.size
        save_debug_image(img_orig_pil, debug_enabled, step_counter_list, debug_dir, "00_original_loaded")

        img_rgb_pil = None
        if img_orig_pil.mode == 'P': # Handle Palette mode explicitly for debug saving
             print(f"Converting image from Palette ('P') to RGB for processing.")
             img_rgb_pil = img_orig_pil.convert('RGB')
             save_debug_image(img_rgb_pil, debug_enabled, step_counter_list, debug_dir, "01_converted_P_to_rgb")
        elif img_orig_pil.mode not in ['RGB', 'RGBA', 'L']: # Allow L mode through for now, convert later if needed
            print(f"Converting image from {img_orig_pil.mode} to RGB for processing.")
            img_rgb_pil = img_orig_pil.convert('RGB')
            save_debug_image(img_rgb_pil, debug_enabled, step_counter_list, debug_dir, "01_converted_to_rgb")
        elif img_orig_pil.mode == 'RGBA':
             print(f"Converting image from RGBA to RGB for core processing.")
             img_rgb_pil = img_orig_pil.convert('RGB')
             save_debug_image(img_rgb_pil, debug_enabled, step_counter_list, debug_dir, "01_converted_rgba_to_rgb")
        else: # Already RGB or L
             img_rgb_pil = img_orig_pil
             save_debug_image(img_rgb_pil, debug_enabled, step_counter_list, debug_dir, f"01_input_is_{img_rgb_pil.mode}")

        # Ensure RGB for stages requiring it
        if img_rgb_pil.mode == 'L':
             print("Converting L mode image to RGB for color processing stages.")
             img_rgb_pil = img_rgb_pil.convert('RGB')
             save_debug_image(img_rgb_pil, debug_enabled, step_counter_list, debug_dir, "02_converted_L_to_rgb")


        # --- Stage 1: Advanced Scaled Quantization and Blending ---
        print("\n--- Stage 1: Running Scaled Multi-Palette Quantization/Blending ---")
        quant_scaled_blended_result = apply_scaled_quantization_and_blend(
            img_rgb_pil, debug_enabled=debug_enabled, step_counter_list=step_counter_list, debug_dir=debug_dir)
        print("--- Stage 1 Completed ---\n")

        # --- Stage 2: EQ, Filtering, Blurring, Recombination (Based on Original RGB) ---
        print("--- Stage 2: Running EQ/Filtering/Recombination ---")
        equalized_img = ImageOps.equalize(img_rgb_pil)
        save_debug_image(equalized_img, debug_enabled, step_counter_list, debug_dir, "stage2_equalized_original")
        equalized_array = np.array(equalized_img).astype(np.float32)

        high_pass_details_array = high_pass_filter(equalized_array, sigma=filter_sigma)

        original_array_float = np.array(img_rgb_pil).astype(np.float32)
        blurred_original_array = gaussian_filter(original_array_float, sigma=(filter_sigma, filter_sigma, 0), mode='nearest', truncate=4.0)
        blurred_original_img = Image.fromarray(np.clip(blurred_original_array, 0, 255).astype(np.uint8), 'RGB')
        save_debug_image(blurred_original_img, debug_enabled, step_counter_list, debug_dir, "stage2_blurred_original")

        filtered_reconstructed_array = blurred_original_array + high_pass_details_array
        filtered_reconstructed_array = np.clip(filtered_reconstructed_array, 0, 255)
        filtered_reconstructed_result = Image.fromarray(filtered_reconstructed_array.astype(np.uint8), 'RGB')
        save_debug_image(filtered_reconstructed_result, debug_enabled, step_counter_list, debug_dir, "STAGE2_RESULT_filtered_reconstructed")
        print("--- Stage 2 Completed ---\n")

        # --- Stage 3: Final Combination ---
        print("--- Stage 3: Blending Stage 1 and Stage 2 Results ---")
        stage3_blend_rgb = Image.blend(quant_scaled_blended_result, filtered_reconstructed_result, filter_blend_alpha)
        save_debug_image(stage3_blend_rgb, debug_enabled, step_counter_list, debug_dir, "STAGE3_RESULT_stage1_stage2_blend")
        print("--- Stage 3 Completed ---\n")

        # --- Stage 4: Post-processing: Alpha / Mode ---
        print("--- Stage 4: Handling Alpha Channel ---")
        final_image_pre_denoise = stage3_blend_rgb
        if original_mode == 'RGBA':
             print("  Adding back alpha channel (fully opaque).")
             alpha = Image.new('L', final_image_pre_denoise.size, 255)
             final_image_pre_denoise.putalpha(alpha)
             save_debug_image(final_image_pre_denoise, debug_enabled, step_counter_list, debug_dir, "stage4_added_alpha")
             # Check output format compatibility later
        else:
            # Image remains RGB (or potentially L if input was L and somehow skipped conversion? Unlikely now)
             print(f"  Image mode is {final_image_pre_denoise.mode}. No alpha added.")
             save_debug_image(final_image_pre_denoise, debug_enabled, step_counter_list, debug_dir, f"stage4_no_alpha_needed_mode_{final_image_pre_denoise.mode}")
        print("--- Stage 4 Completed ---\n")


        # --- Stage 5: Optional Denoising ---
        final_image_output = final_image_pre_denoise # Start with the result from previous stage
        if denoise_strength > 0.0:
            print(f"--- Stage 5: Applying Denoise (Strength: {denoise_strength}) ---")
            print(f"  Applying Median Filter (size=3) to {final_image_output.mode} image...")
            # MedianFilter works directly on RGB/RGBA/L modes in Pillow
            denoised_image = final_image_output.filter(ImageFilter.MedianFilter(size=3))
            save_debug_image(denoised_image, debug_enabled, step_counter_list, debug_dir, "stage5_median_filtered")

            print(f"  Blending original with denoised version (alpha={denoise_strength})...")
            # Blend: im1 * (1.0 - alpha) + im2 * alpha
            # im1 = original (final_image_output), im2 = denoised, alpha = denoise_strength
            final_image_output = Image.blend(final_image_output, denoised_image, alpha=denoise_strength)
            save_debug_image(final_image_output, debug_enabled, step_counter_list, debug_dir, f"STAGE5_RESULT_denoise_blended_{denoise_strength:.2f}")
            print("--- Stage 5 Completed ---\n")
        else:
            print("--- Stage 5: Denoising Skipped (Strength is 0.0) ---\n")


        # --- Saving ---
        print(f"--- Saving Final Image ---")
        print(f"Saving final processed image ({final_image_output.mode}) to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Handle potential RGBA saving issues with JPEG
        save_image = final_image_output
        if output_path.lower().endswith((".jpg", ".jpeg")):
            if save_image.mode == 'RGBA':
                print("Warning: Saving RGBA image as JPEG. Alpha channel will be discarded.")
                save_image = save_image.convert('RGB')
            elif save_image.mode == 'P': # Should not happen here, but safety
                save_image = save_image.convert('RGB')
            elif save_image.mode == 'L':
                 # Allow saving L mode as JPEG
                 pass

            save_image.save(output_path, quality=95, subsampling=0)
        else:
            # Save in a format that supports alpha/other modes (like PNG)
            if original_mode == 'RGBA' and not output_path.lower().endswith(('.png', '.tiff', '.webp')):
                 print(f"Warning: Original mode was RGBA, but output format {os.path.splitext(output_path)[1]} might not fully support it or might lose it. Consider using .png for lossless RGBA.")
            save_image.save(output_path)

        print("\nProcessing finished successfully.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

# --- Command Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply V5 multi-step process (Multi-Palette Quant/Scale/Filter + Denoise + Debug) to potentially disrupt image watermarks.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("output_image", help="Path to save the processed image file.")
    parser.add_argument("--blend_final", type=float, default=0.5, help="Alpha for blend between Stage 1 (Quant/Scale) and Stage 2 (Filter/Reconstruct) results (0.0 to 1.0, default: 0.5).")
    parser.add_argument("--sigma_filter", type=float, default=1.5, help="Sigma value for Gaussian blur/filtering in Stage 2 (default: 1.5).")
    parser.add_argument("--denoise", type=float, default=0.0, help="Strength of final Median denoise blend (0.0 to 1.0, default: 0.0 = off).")
    parser.add_argument("--debug", action='store_true', help="Enable saving intermediate images to a subdirectory.")

    args = parser.parse_args()

    # Argument validation
    if not (0.0 <= args.blend_final <= 1.0):
        print("Error: Final blend alpha must be between 0.0 and 1.0.", file=sys.stderr)
        sys.exit(1)
    if args.sigma_filter <= 0:
        print("Error: Filter sigma must be positive.", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.denoise <= 1.0):
        print("Error: Denoise strength must be between 0.0 and 1.0.", file=sys.stderr)
        sys.exit(1)

    # Run the main processing function
    process_image(args.input_image, args.output_image,
                  filter_blend_alpha=args.blend_final,
                  filter_sigma=args.sigma_filter,
                  debug_enabled=args.debug,
                  denoise_strength=args.denoise) # Pass the denoise strength