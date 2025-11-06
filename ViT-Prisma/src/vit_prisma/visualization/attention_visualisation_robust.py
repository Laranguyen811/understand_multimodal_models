import numpy as np
import torch as t
import os
from typing import List, Any, Callable, Dict, Optional, Tuple, Type, Union, cast
from PIL import Image
from io import BytesIO
import base64
import string, random, json
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from jinja2 import Template

# This function template will be used by the Python script.
# It contains all the HTML, CSS, and JavaScript for the visualisation.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Attention Head Visualisation</title>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f4f8;
            margin: 0;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #334155;
            min-height: 100vh;
        }
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start; /* Change to start for controls */
            gap: 1.5rem;
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            width: 90%;
            max-width: 1400px;
        }
        h1 {
            color: #1e293b;
            margin-bottom: 1.5rem;
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            width: 100%;
        }
        .controls-and-vis {
            display: flex;
            flex-direction: column;
            width: 100%;
            gap: 1rem;
            }
        .vis-container {
            display: flex;
            gap: 2rem;
            justify-content: flex-start;
            flex-wrap: nowrap;
            }
        canvas {
            border: 1px solid #e2e8f0;
            background-color: #fff;
            border-radius: 0.25rem;
            cursor: pointer;
            aspect-ratio: 1 / 1;
            flex-shrink: 0;
            flex-grow: 0;
            width: 50%;
            max-width: 400px;
            }
        .select-container {
            display: flex;
            align-items: center;
            font-weight: 500;
            gap: 0.5rem;
        }
        select {
            padding: 0.5rem;
            border-radius: 0.25rem;
            border: 1px solid #cbd5e1;
            }
        
    </style>
</head>
<body>
    <h1>Multi-Head Attention Visualisation</h1>
    <div class="main-container">
        <div class ="controls-and-vis">
            <div class="select-container">
                <label for="headSelector">Select Attention Head:</label>
                <select id="headSelector"></select>
            </div>
            <div class="vis-container">
                <canvas id="attnCanvas"></canvas>
                <canvas id="imgCanvas"></canvas>
            </div>
        </div>
    </div>
    <script>
        console.log("Starting JavaScript execution...");

        const ATTN_HEADS_DATA = {{ attn_heads_json|safe }};
        const IMAGE_BASE64_DATA = {{ image_base64_json|safe }};
        const NAMES = {{ names_json|safe }};
        const errorMessageDiv = document.getElementById('errorMessage');
        const CLS_TOKEN = {{ cls_token_js }};
        const PATCH_SIZE_IMG = {{ patch_size }};
        const ATTN_SCALING = {{ attn_scaling }};
        const NUM_PATCH_SIDE = {{ num_patch_side }};
        const NUM_TOTAL_TOKENS = {{ num_total_tokens }};
        const ATTN_CANVAS_DIM = NUM_TOTAL_TOKENS * ATTN_SCALING;
        const IMG_CANVAS_DIM = NUM_PATCH_SIDE * PATCH_SIZE_IMG;
        const attnCanvas = document.getElementById('attnCanvas');
        const ctxAttn = attnCanvas.getContext('2d');
        const imgCanvas = document.getElementById('imgCanvas');
        const ctxImg = imgCanvas.getContext('2d');
        const headSelector = document.getElementById('headSelector');
        
        // Canvas setup
        attnCanvas.width = ATTN_CANVAS_DIM;
        attnCanvas.height = ATTN_CANVAS_DIM;
        imgCanvas.width = IMG_CANVAS_DIM;
        imgCanvas.height = IMG_CANVAS_DIM;

        console.log(`NUM_TOTAL_TOKENS: ${NUM_TOTAL_TOKENS}`),
        console.log(`Canvas dimensions:${attnCanvas.width}x${attnCanvas.height}`);
        console.log(`Calculated tokens:${ATTN_CANVAS_DIM / ATTN_SCALING}`);
        console.log(`NUM_PATCH_SIDE: ${NUM_PATCH_SIDE}`);

        // --- Utility Functions (Modified/Shared) ---
        const imageCache = new Map();
        const renderedImageInfoCache = new Map();
        
        // Ensure the onload/onerror are correctly attaached before source setting
        function loadImage(base64Image){
            if (imageCache.has(base64Image)){
                return Promise.resolve(imageCache.get(base64Image)); // A Promise represents a value that does not exist yet but will exist in the future. JS is single-threaded. When we load an image from a base64 string, the decoding and rendering takes time. Without Promises, our code would freeze waiting for the image to load
            }
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    imageCache.set(base64Image, img);
                    resolve(img); // Resolve the promise with the loaded image
                };
                img.onerror = (e) => {
                    console.error("Image failed to load:", e);
                    reject(e);
                }; // Reject the promise if there's an error
                img.src = "data:image/png;base64," + base64Image; // NEW FIX: Add the prefix back here
            });
        }
        function getPatchCoords(tokenIndex) {
            if (CLS_TOKEN && tokenIndex === 0) {
            return null;
            }
            const patchIndex = CLS_TOKEN ? tokenIndex - 1 : tokenIndex;
            const row = Math.floor(patchIndex / NUM_PATCH_SIDE);
            const col = patchIndex % NUM_PATCH_SIDE;
            
            if (row < 0 || row >= NUM_PATCH_SIDE || col < 0 || col >= NUM_PATCH_SIDE) {
                return null;
            }
            return { row, col };
        }
        // Remove NUM_TOTAL_TOKENS
        function drawAttentionMatrix(ctxAttn, attnData) {
            
            if (!attnData) {
                console.error("Attempted to draw with undefined attention data.");
                return;
            }
            // Store the data globally for redrawing during hover
            currentAttnData = attnData;
            ctxAttn.clearRect(0, 0, ctxAttn.canvas.width, ctxAttn.canvas.height);
            const LOG_OFFSET = 1e-8; // Using 1e-6 or 1e-8 works well for floating point data
            
            // Absolute Log Minimum: This is the floor of our visualisation
            const ABSOLUTE_LOG_MIN = Math.log10(LOG_OFFSET); // log10(1e-8) = -8
            /*const MIN_LOG_RANGE = 8.0;*/ // The minimum log-range (Max - Min) to enforce contrast. 

            // First Pass: Calculate Dynamic Log Min/Max
            // let currentLogMin = Infinity; // Start high
            let currentLogMax = -Infinity; // Start low

            for (let r = 0; r < NUM_TOTAL_TOKENS; r++) {
            if (r === 0) console.log("Attention Draw: Starting row 0. NUM_TOTAL_TOKENS:", NUM_TOTAL_TOKENS);
                if (!attnData[r]) continue;
                for (let c = 0; c < NUM_TOTAL_TOKENS; c++) {
                    
                    const rawIntensity = attnData[r][c];
                    const intensity = typeof rawIntensity === 'number' ? rawIntensity : 0; 
                    // Apply log normalisation. Use a small constant (1e-10) to prevent log(0) which is -Infinity
                    const logValue = Math.log10(intensity + LOG_OFFSET);
                    // currentLogMin = Math.min(currentLogMin, logValue);
                    currentLogMax = Math.max(currentLogMax, logValue);
                }
            }
            const effectiveLogMin = ABSOLUTE_LOG_MIN; // Low values will be clamped here
            const effectiveLogMax = currentLogMax; // High values use the actual max found 
            const logRange = effectiveLogMax - effectiveLogMin;
            console.log(`minLog: ${effectiveLogMin}, maxLog: ${effectiveLogMax}, logRange: ${logRange}`);
            
            // Second Pass: Draw the Pixels 
            const isMeaningfulRange = logRange > 0.0001; // Prevent division by zero or near-zero
            for (let r = 0; r < NUM_TOTAL_TOKENS; r++){
            if (!attnData[r]) continue;
                for (let c = 0; c < NUM_TOTAL_TOKENS; c++){
                    const rawIntensity = attnData[r][c];
                    const intensity = typeof rawIntensity === 'number' ? rawIntensity : 0;
            
                    // Re-calculate log value and normalise using the dynamic range
                    const logIntensity = Math.log10(intensity + LOG_OFFSET);
                    

                    // The numerator must use the clamped value to prevent negative intensity
                    let normalisedIntensity = 0;
                    if (isMeaningfulRange) {
                    // Clamp to ensure logIntensity is not below the defined floor (effectiveLogMin)
                        const clampedLogIntensity = Math.max(logIntensity, effectiveLogMin);
                    
                    // Normalise: map clamped value from [effectiveLogMin, effectiveLogMax] to [0, 1]
                        normalisedIntensity = (clampedLogIntensity - effectiveLogMin) / logRange;
                    }

                    // Add power transform for more contrast
                    normalisedIntensity = Math.pow(normalisedIntensity, 0.5);

                    // HSL: Use the wide-range settings for max visibility
                    // HUE: 250 (Purple/Low) -> 50 (Yellow/High)
                    const hue = 250 - (normalisedIntensity * 200);
                    const saturation = 100;
                    // LIGHTNESS: 20% (Dark/Min) to 90% (Bright/Max)
                    const lightness = 20 + (normalisedIntensity * 70);
                    if (r === 0 && c < 5) {
                        console.log(`Pixel [0, ${c}]: intensity=${intensity}, logIntensity=${logIntensity}, normalised=${normalisedIntensity}, hue=${hue}`)
                    }
                    ctxAttn.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                    ctxAttn.fillRect(c * ATTN_SCALING, r * ATTN_SCALING, ATTN_SCALING, ATTN_SCALING);
                    }
                }
            
            // Draw grid lines for clarity
            ctxAttn.strokeStyle = 'rgba(255, 255, 255, 0.5)'; // White lines, semi-transparent
            ctxAttn.lineWidth = 1;

            // Vertical lines
            for (let i = 0; i <= NUM_TOTAL_TOKENS; i++) {
                const x = i * ATTN_SCALING;
                ctxAttn.beginPath();
                ctxAttn.moveTo(x, 0);
                ctxAttn.lineTo(x, ctxAttn.canvas.height);
                ctxAttn.stroke();
            }
            // Horizontal lines
            for (let i = 0; i <= NUM_TOTAL_TOKENS; i++) {
                const y = i * ATTN_SCALING;
                ctxAttn.beginPath();
                ctxAttn.moveTo(0, y);
                ctxAttn.lineTo(ctxAttn.canvas.width, y);
                ctxAttn.stroke();
            }
        }
        
        // Draw a white square highlight on the attention matrix
        function drawAttnHighlight(ctxAttn, queryTokenIndex, keyTokenIndex) {
            ctxAttn.save();
            const x = keyTokenIndex * ATTN_SCALING; // X-axis = Key index
            const y = queryTokenIndex * ATTN_SCALING; // Y-axis = Query index

            // Draw a solid white fill
            ctxAttn.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctxAttn.fillRect(x, y, ATTN_SCALING, ATTN_SCALING);

            // Redraw a darker border on the highlighted square 
            ctxAttn.strokeStyle = 'rgba(0, 0, 0, 0.8)';
            ctxAttn.lineWidth = 1;
            ctxAttn.strokeRect(x, y, ATTN_SCALING, ATTN_SCALING);

            ctxAttn.restore();
        }
    
        // Draws the initial image + patch grid
        function drawImageAndGrid(ctxImg, base64Image) {
            const canvasW = ctxImg.canvas.width;
            const canvasH = ctxImg.canvas.height;

            loadImage(base64Image).then((img) => {
                ctxImg.clearRect(0, 0, canvasW, canvasH);
                // Draw at natural size instead of forcing canvas dimensions
                const imgW = img.width;
                const imgH = img.height;

                const scale = Math.min(canvasW / imgW, canvasH / imgH);
                const w = imgW * scale;
                const h = imgH * scale;
                const x = (canvasW - w) / 2;
                const y = (canvasH - h) / 2;
                
                // Store successful image info
                renderedImageInfoCache.set("current",
                {
                img: img,
                scale: scale,
                offsetX: x,
                offsetY :y, 
                renderedW: w,
                renderedH : h,
                isFallback: false
                });
    

                ctxImg.drawImage(img, x, y, w, h);

                // Draw the fixed patch grid overlay
                drawPatchGrid(ctxImg, scale, x, y, w, h);
            }).catch(e => { console.warn("Failed to load image, drawing fallback rectangle instead:", e);
            
                // Fallback if image fails to load
                
                renderedImageInfoCache.set("current",
                {
                img: img,
                scale: 1,
                offsetX: 0,
                offsetY :0, 
                renderedW: canvasW,
                renderedH : canvasH,
                isFallback: true
                });

                // Draw simple gray background
                ctxImg.clearRect(0, 0, canvasW, canvasH);
                ctxImg.fillStyle = '#e2e8f0'; // Light gray
                ctxImg.fillRect(0, 0, canvasW, canvasH);

                // Draw the patch grid overlay
                drawPatchGrid(ctxImg, 1, 0, 0, canvasW, canvasH);

                // Draw text for user feedback
                ctxImg.fillStyle = '#666';
                ctxImg.font = '10px Inter';
                ctxImg.textAlign = 'center';
                ctxImg.fillText('Image Load Error', canvasW / 2, canvasH / 2 - 5);
                ctxImg.fillText('Using Fallback', canvasW / 2, canvasH / 2 + 10);
                });
            }

        function drawPatchGrid(ctxImg, scale, offsetX, offsetY, renderedW, renderedH, highlightedPatches = null) {
            ctxImg.save();
            const patchDim = PATCH_SIZE_IMG * scale;
            // Draw patch backgrounds if highlighted 
            if (highlightedPatches) {
                for (let row = 0; row < NUM_PATCH_SIDE; row++) {
                    for (let col = 0; col < NUM_PATCH_SIDE; col++) {
                        const patchIndex = row * NUM_PATCH_SIDE + col;
                        const tokenIndex = patchIndex + (CLS_TOKEN ? 1 : 0);

                        if (highlightedPatches[tokenIndex]) {
                            const intensity = highlightedPatches[tokenIndex];
                            const x = offsetX + col * patchDim;
                            const y = offsetY + row * patchDim;
                            // Color based on intensity
                            // HUE: 250 (Purple/Low) -> 50 (Yellow/High)
                            const alpha = intensity * 0.7;
                            ctxImg.fillStyle = `rgba(255, 255, 0, ${alpha})`; // Yellow overlay
                            ctxImg.fillRect(x, y, patchDim, patchDim); 
                        }
                    }
                }
            }
            

            ctxImg.strokeStyle = 'rgba(0, 0, 0, 0.6)';
            ctxImg.lineWidth = 1;
            // Vertical Lines
            for (let i = 0; i <= NUM_PATCH_SIDE; i++) {
                const x = offsetX + i * patchDim;
                ctxImg.beginPath();
                ctxImg.moveTo(x, offsetY);
                ctxImg.lineTo(x, offsetY + renderedH);
                ctxImg.stroke();
            }

            // Horizontal Lines
            for (let i = 0; i <= NUM_PATCH_SIDE; i++) {
                const y = offsetY + i * patchDim;
                ctxImg.beginPath();
                ctxImg.moveTo(offsetX, y);
                ctxImg.lineTo(offsetX + renderedW, y);
                ctxImg.stroke();
            }
            ctxImg.restore();
        }

        function drawPatchHighlight(ctx, tokenIndex, color, type) {
            const info = renderedImageInfoCache.get("current");
            const coords = getPatchCoords(tokenIndex);
            if (!info || !coords) return;

            const patchDim = PATCH_SIZE_IMG * info.scale;
            const x = info.offsetX + coords.col * patchDim;
            const y = info.offsetY + coords.row * patchDim;

            ctx.save();
            ctx.fillStyle = type === 'query' ? 'rgba(16, 185, 129, 0.4)' : 'rgba(236, 72, 153, 0.4)';
            ctx.fillRect(x, y, patchDim, patchDim);

            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, patchDim, patchDim);
            ctx.restore();
        }

        function redrawQueryAttention(ctxImg, attnData, queryTokenIndex, base64Image) {
            const info = renderedImageInfoCache.get("current");
            if (!info) return;

            const attnScores = attnData[queryTokenIndex];
            if (!attnScores) return;

            ctxImg.clearRect(0, 0, ctxImg.canvas.width, ctxImg.canvas.height);
            if (!info.isFallback) {
            ctxImg.drawImage(info.img, info.offsetX, info.offsetY, info.renderedW, info.renderedH);
            } else {
            ctxImg.fillStyle = '#e2e8f0'; // Light gray
            ctxImg.fillRect(0, 0, ctxImg.canvas.width, ctxImg.canvas.height);
            ctxImg.fillStyle = '#666';
            ctxImg.font = '10px Inter';
            ctxImg.textAlign = 'center';
            ctxImg.fillText('Image Load Error', ctxImg.canvas.width / 2, ctxImg.canvas.height / 2 - 5);
            ctxImg.fillText('Using Fallback', ctxImg.canvas.width / 2, ctxImg.canvas.height / 2 + 10);
            }

            // Create highlighted patches map
            const highlightMap = {};
            for (let i = 0; i < attnScores.length; i++) {
                if (attnScores[i] > 0) {
                    highlightMap[i] = attnScores[i];
                }
            }
            drawPatchGrid(ctxImg, info.scale, info.offsetX, info.offsetY, info.renderedW, info.renderedH, highlightMap);
            drawPatchHighlight(ctxImg, queryTokenIndex,"#10b981", "query")
        }

        function redrawWithPairedHighlights(ctxImg, attnData, queryTokenIndex, keyTokenIndex, base64Image) {
            const info = renderedImageInfoCache.get("current");
            if (!info) return;

            const attnScores = attnData[queryTokenIndex];
            if (!attnScores) return;

            ctxImg.clearRect(0, 0, ctxImg.canvas.width, ctxImg.canvas.height);
            if (!info.isFallback) {
            ctxImg.drawImage(info.img, info.offsetX, info.offsetY, info.renderedW, info.renderedH);
            }
            else {
            ctxImg.fillStyle = '#e2e8f0'; // Light gray
            ctxImg.fillRect(0, 0, ctxImg.canvas.width, ctxImg.canvas.height);
            ctxImg.fillStyle = '#666';
            ctxImg.font = '10px Inter';
            ctxImg.textAlign = 'center';
            ctxImg.fillText('Image Load Error', ctxImg.canvas.width / 2, ctxImg.canvas.height / 2 - 5);
            ctxImg.fillText('Using Fallback', ctxImg.canvas.width / 2, ctxImg.canvas.height / 2 + 10);
            }

            // Create highlighted patches map (exclude query and key for cleaner visual)
            const highlightMap ={};
            for (let i = 0; i < attnScores.length; i++) {
                if (attnScores[i] > 0 && i !== queryTokenIndex && i !== keyTokenIndex) {
                    highlightMap[i] = attnScores[i] * 0.4; // Slightly dimmer for context
 
                }
            }
        

            drawPatchGrid(ctxImg, info.scale, info.offsetX, info.offsetY, info.renderedW, info.renderedH, highlightMap);

            const patchDim = PATCH_SIZE_IMG * info.scale;

            drawPatchHighlight(ctxImg, queryTokenIndex, '#10b981', 'query');
            drawPatchHighlight(ctxImg, keyTokenIndex, '#ec4899', 'key');

            const qCoords = getPatchCoords(queryTokenIndex);
            const kCoords = getPatchCoords(keyTokenIndex);

            console.log(`Hover: query=${queryTokenIndex}, key=${keyTokenIndex}`);
            console.log(`Query coords: row=${qCoords?.row}, col=${qCoords?.col}`);
            console.log(`Key coords: row=${kCoords?.row}, col=${kCoords?.col}`);

            //if (qCoords && kCoords && queryTokenIndex !== keyTokenIndex) {
              //  const qx = info.offsetX + (qCoords.col + 0.5) * patchDim;
              //  const qy = info.offsetY + (qCoords.row + 0.5) * patchDim;
              //  const kx = info.offsetX + (kCoords.col + 0.5) * patchDim;
              //  const ky = info.offsetY + (kCoords.row + 0.5) * patchDim;

              //  const score = attnScores[keyTokenIndex] || 0;
              //  const thickness = 1 + score * 5;

              //  ctxImg.save();
              //  ctxImg.strokeStyle = 'rgba(255, 255, 255, 0.8)';
              //  ctxImg.lineWidth = thickness + 2;
              //  ctxImg.beginPath();
              //  ctxImg.moveTo(qx, qy);
              //  ctxImg.lineTo(kx, ky);
              //  ctxImg.stroke();

              //  ctxImg.strokeStyle = 'rgba(255, 0, 0, 1)';
              //  ctxImg.lineWidth = thickness;
              //  ctxImg.beginPath();
              //  ctxImg.moveTo(qx, qy);
              //  ctxImg.lineTo(kx, ky);
              //  ctxImg.stroke();
              //  ctxImg.restore();       
            

        }

        // Core Visualisation Logic

        function drawVisualisation(headIndex) {
            if (!ATTN_HEADS_DATA || headIndex < 0 || headIndex >= ATTN_HEADS_DATA.length) return;

            const attnData = ATTN_HEADS_DATA[headIndex];
            const headImage = IMAGE_BASE64_DATA[headIndex];

            // Draw Attention Matrix
            drawAttentionMatrix(ctxAttn, attnData);

            // Draw Base Image and Grid
            drawImageAndGrid(ctxImg, headImage);

            // Re-attach Event Listeners (ensure they reference the correct data for the active head)

            // Mouse hover over Attention Matrix (Query Token = row index)
            attnCanvas.onmousemove = (e) => {
                const rect = attnCanvas.getBoundingClientRect(); // return a DOMRect object for info about the size of an element and its position relative to the viewpoint 
                const borderWidth = 1;
                const scaleX = attnCanvas.width / rect.width;
                const scaleY = attnCanvas.height / rect.height;
                
                const mouseX = (e.clientX - rect.left - borderWidth)  * (attnCanvas.width / (rect.width - 2 * borderWidth));
                const mouseY = (e.clientY - rect.top - borderWidth)  * (attnCanvas.height / (rect.height - 2 * borderWidth));

                const queryTokenIndex = Math.floor(mouseY / ATTN_SCALING);
                const keyTokenIndex = Math.floor(mouseX / ATTN_SCALING);

                console.log(`Hover: query=${queryTokenIndex}, key=${keyTokenIndex}`);
                console.log(`Patch coords: query=${getPatchCoords(queryTokenIndex)}, key=${getPatchCoords(keyTokenIndex)}`);
                
                if (queryTokenIndex >= 0 && queryTokenIndex < NUM_TOTAL_TOKENS &&
                    keyTokenIndex >= 0 && keyTokenIndex < NUM_TOTAL_TOKENS) {
                    // Redraw the entire matrix to remove previous highlights
                    drawAttentionMatrix(ctxAttn, attnData);
                    // Draw highlight on the hovered cell
                    drawAttnHighlight(ctxAttn, queryTokenIndex, keyTokenIndex);

                    // Draw image with paired highlights
                    redrawWithPairedHighlights(ctxImg, attnData, queryTokenIndex, keyTokenIndex, headImage);
                    } 

                };
                // Mouse leaves Attention Matrix
                attnCanvas.onmouseout = () => {
                if (currentAttnData) {
                // Redraw the matrix one last time to remove highlight
                    drawAttentionMatrix(ctxAttn, currentAttnData);
                }
                    // Clear image highlights                
                    drawImageAndGrid(ctxImg, headImage);
            };

            // Mouse hover over Image (Query Token = patch index + CLS token offset)
            imgCanvas.onmousemove = (e) => {
                const info = renderedImageInfoCache.get("current");
                if (!info) return;

                const rect = imgCanvas.getBoundingClientRect();
                const clickXRel = e.clientX - rect.left - info.offsetX;
                const clickYRel = e.clientY - rect.top - info.offsetY;

                if (clickXRel < 0 || clickYRel < 0 || clickXRel > info.renderedW || clickYRel > info.renderedH) {
                    drawImageAndGrid(ctxImg, headImage);
                    return;
                    }
                    const patchDim = PATCH_SIZE_IMG * info.scale;
                    const x = Math.floor(clickXRel / patchDim);
                    const y = Math.floor(clickYRel / patchDim);
                    const patchIndex = y * NUM_PATCH_SIDE + x;
                    const tokenIndex = patchIndex + (CLS_TOKEN ? 1 :0);
                    
                    // Highlight the image patch
                    redrawQueryAttention(ctxImg, attnData, tokenIndex, headImage);
                    
                    // Highlight the corresponding row in the attention matrix
                    if (currentAttnData) {
                        drawAttentionMatrix(ctxAttn, currentAttnData);
                        // Draw white highlight across the entire row corresponding to the query token
                        ctxAttn.fillStyle = 'rgba(255, 255, 255, 0.2)';
                        ctxAttn.fillRect(0, tokenIndex * ATTN_SCALING, ctxAttn.canvas.width, ATTN_SCALING);
                    }
                };
            
            
            imgCanvas.onmouseout = () => {
                drawImageAndGrid(ctxImg, headImage);
                if (currentAttnData) {
                    drawAttentionMatrix(ctxAttn, currentAttnData);
                }
            };
        }
    
        function setupSelector() {
            if (!NAMES || NAMES.length === 0) return;
            headSelector.innerHTML = '';
            NAMES.forEach((name, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = name;
                headSelector.appendChild(option);
            });

            headSelector.onchange = () => {
                const selectedIndex = parseInt(headSelector.value);
                drawVisualisation(selectedIndex);
            };

            drawVisualisation(0);

            }    
        window.onload = setupSelector;

    </script>
</body>
</html>
"""

def convert_to_3_channels(image):
    """ 
    Convert images to 3-channel images. 
    Handles both [H,W,1] and [H,W] formats.
    """
    if image.ndim == 2:
        # If image is 2D (grayscale), stack it 3 times 
        return np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        # if the image has a single color channel, squeeze it and then stack
        squeezed_image = np.squeeze(image, axis=-1)
        return np.stack([squeezed_image, squeezed_image, squeezed_image], axis=-1)
    return image

def prepare_image(image):
    """ 
    Prepare images for proper formatting of (C,H,W).
    """
    if isinstance(image, t.Tensor):
        image = image.numpy()

    # Check if the image needs transposing from [C, H, W] to [H, W, C]
    # It assumes that if a 3D image has a first dimension of 1 or 3, it is a channel dimension. 
    # Handle channel-first format
    if image.ndim == 3 and (image.shape[0] == 3 or image.shape[0] == 1):
        # Channel-first to channel-last
            image = np.transpose(image, (1, 2, 0))
    # Ensure the image has 3 channels for display, regardless of its previous state
        # Normalise pixel values to the 0-255 range and convert to uint8
    image = (image - image.min()) / (image.max() - image.min()) * 255 # Preserve floating-point precision, display requirements and be consistent for convert_to_3_channels
    image = image.astype('uint8')
    image = convert_to_3_channels(image)
    return image

def generate_random_string(length=10):
    """
    Helper function to generate canvas IDs for javascript figures.
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def image_to_base64(img_array: np.ndarray) -> str:
    """ Change images' resolutions to base64."""
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") # Unicode Transformation Format - 8 bits
    return img_str

def norm_attn_head(attn_head):
    """ Normalise attention heads."""
    min_val = np.min(attn_head)
    max_val = np.max(attn_head)
    if (max_val - min_val) == 0:
        return attn_head # Avoid division by 0
    normalised_attn_head = (attn_head - min_val) / (max_val - min_val)
    return normalised_attn_head

# Prepare data to send to javascript
class AttentionHeadImageJSInfo:
    """
    Prepares attention and image data for JavaScript visualisation.
    """
    def __init__(self, attn_head, cfg, image, name="No Name", cls_token=True):
        # NEW: Robustly handle 1D arrays by reshaping them
        if attn_head.ndim == 1:
            print(f"Warning: Reshaping a 1D attention head array of size {attn_head.shape[0]} into a square matrix.")
            size = int(np.sqrt(attn_head.shape[0]))
            if size * size == attn_head.shape[0]:
                attn_head = attn_head.reshape(size, size)

            else:
                print(f"Error: Could not reshape 1D array of size {attn_head.shape[0]} into a square matrix.")
                attn_head = np.zeros((1, 1)) # Fallback to a single zero element
        elif attn_head.ndim > 2:
            print("Warning: Reshaping a >2D attention head array.")
            # Flatten to a 2D array, assuming the first dimension is a batch or an extra dimension
            attn_head = np.squeeze(attn_head,axis=0)  
        print(f"Shape of attention head: {attn_head.ndim}")  
        
        #normalised_ah = norm_attn_head(attn_head)

        self.attn_head = attn_head.tolist()
        self.name = name 

        # The image size should come from the model configuration
        patch_size = cfg.patch_size
        image_size = cfg.image_size

        print(f"Image shape before prepare_image:{image.shape}")
        print(f"Image shape after prepare_image:{prepare_image(image).shape}")
        print(f"Expected image shape: (224,224,3)")
        # Prepare the image for base64 encoding
        self.image_base64 = image_to_base64(prepare_image(image))

def generate_html_and_js_code(
    attn_heads_json, 
    image_base64_json, 
    names_json, 
    ATTN_SCALING, 
    cls_token, 
    patch_size, 
    num_patch_side, 
    num_total_tokens
):
    # This function now correctly renders the Jinja2 template.
    template = Template(HTML_TEMPLATE)
    rendered_html = template.render(
        attn_heads_json=attn_heads_json,
        image_base64_json=image_base64_json,
        names_json=names_json,
        attn_scaling=ATTN_SCALING,
        cls_token_js=str(cls_token).lower(),
        patch_size=patch_size,
        num_patch_side=num_patch_side,
        num_total_tokens=num_total_tokens
    )
    return rendered_html

def plot_javascript(
        list_of_attn_heads: Union[t.Tensor, List[np.ndarray]],
        list_of_images: Union[List[np.ndarray], np.ndarray],
        cfg,
        list_of_names: Optional[Union[t.Tensor, List[str]]] = None,
        ATTN_SCALING: int = 8,
        cls_token: bool = True
    ):
    if isinstance(list_of_attn_heads, t.Tensor):
        list_of_attn_heads = [np.array(list_of_attn_heads[i]) for i in range(list_of_attn_heads.shape[0])]
    elif not isinstance(list_of_attn_heads, list):
        list_of_attn_heads = [list_of_attn_heads]
    
    if not isinstance(list_of_images, list):
        list_of_images = [list_of_images]
    
    if isinstance(list_of_names, t.Tensor):
        list_of_names = [str(i) for i in list_of_names.tolist()]
    if list_of_names is None:
        list_of_names = [f"Attention Head {i+1}" for i in range(len(list_of_attn_heads))]

    assert len(list_of_attn_heads) == len(list_of_images), "Must provide an image for each attention head"
    assert len(list_of_attn_heads) == len(list_of_names), "Must provide a name for each attention head"

    attn_head_image_js_infos = []
    for attn_head, image, name in zip(list_of_attn_heads, list_of_images, list_of_names):
        attn_head_image_js_infos.append(AttentionHeadImageJSInfo(attn_head, cfg, image, name=name, cls_token=cls_token))
    
    # Prepare data for JavaScript
    attn_heads_json = json.dumps([info.attn_head for info in attn_head_image_js_infos])
    image_base64_json = json.dumps([info.image_base64 for info in attn_head_image_js_infos])
    names_json = json.dumps([info.name for info in attn_head_image_js_infos])

    # Correct calculation of number of patches
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    num_patch_side = int(np.sqrt(num_patches))
    num_total_tokens = num_patches + (1 if cls_token else 0)
    
    # --- ADDED DEBUGGING PRINTS ---
    print("--- DEBUGGING DATA ---")

    if list_of_attn_heads and list_of_images:
        print(f"Number of attention heads: {len(list_of_attn_heads)}")
        print(f"First 10 values of the first attention head:{list_of_attn_heads[0].flatten()[:10]}")
        print(f"First 10 values of the first image data: {list_of_images[0].flatten()[:10]}")
    else:
        print("Input lists are empty or invalid.")  
    
    print("--- END DEBUGGING DATA ---")
    # ----------------------------
    # Now, call the new generation function and most importantly, RETURN its result
    return generate_html_and_js_code(
        attn_heads_json=attn_heads_json,
        image_base64_json=image_base64_json,
        names_json=names_json,
        ATTN_SCALING=ATTN_SCALING,
        cls_token=cls_token,
        patch_size=cfg.patch_size,
        num_patch_side=num_patch_side,
        num_total_tokens=num_total_tokens
    )
