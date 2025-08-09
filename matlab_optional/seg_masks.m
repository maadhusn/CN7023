% seg_masks.m - MATLAB script for generating segmentation masks
% This is an optional component for advanced preprocessing

function seg_masks(input_dir, output_dir, variant)
    % Generate segmentation masks for plant disease images
    %
    % Args:
    %   input_dir: Path to input images directory
    %   output_dir: Path to output masks directory  
    %   variant: Dataset variant ('color', 'grayscale', 'segmented')
    
    if nargin < 3
        variant = 'color';
    end
    
    fprintf('Generating segmentation masks...\n');
    fprintf('Input directory: %s\n', input_dir);
    fprintf('Output directory: %s\n', output_dir);
    fprintf('Variant: %s\n', variant);
    
    % Create output directory structure
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Get list of class directories
    class_dirs = dir(fullfile(input_dir, variant));
    class_dirs = class_dirs([class_dirs.isdir] & ~ismember({class_dirs.name}, {'.', '..'}));
    
    total_processed = 0;
    
    for i = 1:length(class_dirs)
        class_name = class_dirs(i).name;
        class_input_dir = fullfile(input_dir, variant, class_name);
        class_output_dir = fullfile(output_dir, variant, class_name);
        
        % Create class output directory
        if ~exist(class_output_dir, 'dir')
            mkdir(class_output_dir);
        end
        
        fprintf('Processing class: %s\n', class_name);
        
        % Get list of images
        image_files = dir(fullfile(class_input_dir, '*.jpg'));
        image_files = [image_files; dir(fullfile(class_input_dir, '*.png'))];
        
        for j = 1:length(image_files)
            image_name = image_files(j).name;
            [~, base_name, ~] = fileparts(image_name);
            
            image_path = fullfile(class_input_dir, image_name);
            mask_path = fullfile(class_output_dir, [base_name, '.png']);
            
            % Generate mask
            mask = generate_plant_mask(image_path);
            
            % Save mask
            imwrite(mask, mask_path);
            
            total_processed = total_processed + 1;
            
            if mod(j, 10) == 0
                fprintf('  Processed %d/%d images\n', j, length(image_files));
            end
        end
    end
    
    fprintf('Segmentation mask generation completed!\n');
    fprintf('Total images processed: %d\n', total_processed);
end

function mask = generate_plant_mask(image_path)
    % Generate binary mask for plant regions
    %
    % Args:
    %   image_path: Path to input image
    %
    % Returns:
    %   mask: Binary mask (logical array)
    
    % Read image
    img = imread(image_path);
    
    % Convert to different color spaces for better segmentation
    if size(img, 3) == 3
        % RGB to HSV
        hsv = rgb2hsv(img);
        h = hsv(:,:,1);
        s = hsv(:,:,2);
        v = hsv(:,:,3);
        
        % RGB to Lab
        lab = rgb2lab(img);
        a = lab(:,:,2);
        b = lab(:,:,3);
        
        % Green vegetation mask using HSV
        green_mask = (h >= 0.2 & h <= 0.4) & (s >= 0.2) & (v >= 0.1);
        
        % Refine using Lab color space (green regions have negative 'a' values)
        green_mask = green_mask & (a < 10);
        
    else
        % Grayscale - use intensity thresholding
        green_mask = img > graythresh(img) * 255;
    end
    
    % Morphological operations to clean up the mask
    se = strel('disk', 3);
    mask = imopen(green_mask, se);  % Remove small noise
    mask = imclose(mask, se);       % Fill small gaps
    
    % Remove small connected components
    mask = bwareaopen(mask, 100);
    
    % Fill holes
    mask = imfill(mask, 'holes');
    
    % Convert to uint8 for saving
    mask = uint8(mask) * 255;
end
