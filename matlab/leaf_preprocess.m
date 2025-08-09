function leaf_preprocess(root, out, variant)
% LEAF_PREPROCESS Preprocess plant images with HSV thresholding and morphology
%
% Inputs:
%   root - Input dataset path (e.g., 'C:\PlantVillage')
%   out - Output processed dataset path (e.g., 'C:\PlantVillage_processed')
%   variant - Dataset variant ('color', 'grayscale', etc.)
%
% Processing steps:
%   1. Read RGB image
%   2. Convert to HSV color space
%   3. Apply S-channel thresholding for vegetation
%   4. Morphological opening and closing
%   5. Keep largest connected component
%   6. Histogram equalization on V channel
%   7. Save to mirrored output path

if nargin < 3
    variant = 'color';
end

% Get all class directories
input_dir = fullfile(root, variant);
output_dir = fullfile(out, variant);

if ~exist(input_dir, 'dir')
    error('Input directory does not exist: %s', input_dir);
end

% Create output directory structure
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Get class folders
class_dirs = dir(input_dir);
class_dirs = class_dirs([class_dirs.isdir] & ~ismember({class_dirs.name}, {'.', '..'}));

fprintf('Processing %d classes...\n', length(class_dirs));

total_processed = 0;
samples_per_class = 50; % Process ~50 images per class for demonstration

for i = 1:length(class_dirs)
    class_name = class_dirs(i).name;
    class_input_dir = fullfile(input_dir, class_name);
    class_output_dir = fullfile(output_dir, class_name);
    
    % Create output class directory
    if ~exist(class_output_dir, 'dir')
        mkdir(class_output_dir);
    end
    
    % Get image files
    img_files = dir(fullfile(class_input_dir, '*.jpg'));
    if isempty(img_files)
        img_files = [dir(fullfile(class_input_dir, '*.jpeg')); ...
                     dir(fullfile(class_input_dir, '*.png'))];
    end
    
    % Process up to samples_per_class images
    num_to_process = min(length(img_files), samples_per_class);
    
    fprintf('Processing class %s: %d/%d images\n', class_name, num_to_process, length(img_files));
    
    for j = 1:num_to_process
        try
            % Read image
            img_path = fullfile(class_input_dir, img_files(j).name);
            rgb_img = imread(img_path);
            
            % Convert to HSV
            hsv_img = rgb2hsv(rgb_img);
            
            % Extract channels
            h_channel = hsv_img(:,:,1);
            s_channel = hsv_img(:,:,2);
            v_channel = hsv_img(:,:,3);
            
            % Threshold S channel for vegetation (green areas typically have high saturation)
            s_thresh = 0.3; % Adjust threshold as needed
            vegetation_mask = s_channel > s_thresh;
            
            % Morphological operations
            se = strel('disk', 3); % Structuring element
            
            % Opening (erosion followed by dilation) - removes noise
            mask_opened = imopen(vegetation_mask, se);
            
            % Closing (dilation followed by erosion) - fills holes
            mask_closed = imclose(mask_opened, se);
            
            % Keep largest connected component
            cc = bwconncomp(mask_closed);
            if cc.NumObjects > 0
                areas = cellfun(@numel, cc.PixelIdxList);
                [~, largest_idx] = max(areas);
                final_mask = false(size(mask_closed));
                final_mask(cc.PixelIdxList{largest_idx}) = true;
            else
                final_mask = mask_closed; % Fallback if no components found
            end
            
            % Apply mask to original image
            masked_rgb = rgb_img;
            for ch = 1:3
                channel = masked_rgb(:,:,ch);
                channel(~final_mask) = 0; % Set background to black
                masked_rgb(:,:,ch) = channel;
            end
            
            % Histogram equalization on V channel of masked region
            if any(final_mask(:))
                % Convert masked image to HSV
                masked_hsv = rgb2hsv(masked_rgb);
                v_masked = masked_hsv(:,:,3);
                
                % Apply histogram equalization only to vegetation pixels
                v_eq = v_masked;
                vegetation_pixels = v_masked(final_mask);
                if ~isempty(vegetation_pixels) && max(vegetation_pixels) > min(vegetation_pixels)
                    eq_pixels = histeq(vegetation_pixels);
                    v_eq(final_mask) = eq_pixels;
                end
                
                % Reconstruct HSV and convert back to RGB
                masked_hsv(:,:,3) = v_eq;
                final_rgb = hsv2rgb(masked_hsv);
                final_rgb = im2uint8(final_rgb);
            else
                final_rgb = masked_rgb;
            end
            
            % Save processed image
            [~, name, ext] = fileparts(img_files(j).name);
            output_path = fullfile(class_output_dir, [name, '.jpg']);
            imwrite(final_rgb, output_path, 'Quality', 95);
            
            total_processed = total_processed + 1;
            
        catch ME
            fprintf('Error processing %s: %s\n', img_files(j).name, ME.message);
        end
    end
end

fprintf('Preprocessing completed! Processed %d images total.\n', total_processed);
fprintf('Output saved to: %s\n', output_dir);
fprintf('\nTo use processed dataset, update config.yaml:\n');
fprintf('dataset:\n  path: "%s"\n', out);

end
