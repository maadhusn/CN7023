% extract_feats.m - MATLAB script for extracting handcrafted features
% This is an optional component for traditional ML approaches

function features = extract_feats(input_dir, output_file, variant)
    % Extract handcrafted features from plant disease images
    %
    % Args:
    %   input_dir: Path to input images directory
    %   output_file: Path to save extracted features (.mat file)
    %   variant: Dataset variant ('color', 'grayscale', 'segmented')
    %
    % Returns:
    %   features: Structure containing extracted features and labels
    
    if nargin < 3
        variant = 'color';
    end
    
    fprintf('Extracting handcrafted features...\n');
    fprintf('Input directory: %s\n', input_dir);
    fprintf('Output file: %s\n', output_file);
    fprintf('Variant: %s\n', variant);
    
    % Initialize feature storage
    all_features = [];
    all_labels = [];
    class_names = {};
    
    % Get list of class directories
    class_dirs = dir(fullfile(input_dir, variant));
    class_dirs = class_dirs([class_dirs.isdir] & ~ismember({class_dirs.name}, {'.', '..'}));
    
    for i = 1:length(class_dirs)
        class_name = class_dirs(i).name;
        class_dir = fullfile(input_dir, variant, class_name);
        class_names{end+1} = class_name;
        
        fprintf('Processing class %d/%d: %s\n', i, length(class_dirs), class_name);
        
        % Get list of images
        image_files = dir(fullfile(class_dir, '*.jpg'));
        image_files = [image_files; dir(fullfile(class_dir, '*.png'))];
        
        class_features = [];
        
        for j = 1:length(image_files)
            image_path = fullfile(class_dir, image_files(j).name);
            
            % Extract features from single image
            img_features = extract_single_image_features(image_path);
            
            class_features = [class_features; img_features];
            
            if mod(j, 20) == 0
                fprintf('  Processed %d/%d images\n', j, length(image_files));
            end
        end
        
        % Store features and labels
        all_features = [all_features; class_features];
        all_labels = [all_labels; repmat(i, size(class_features, 1), 1)];
    end
    
    % Create features structure
    features = struct();
    features.data = all_features;
    features.labels = all_labels;
    features.class_names = class_names;
    features.feature_names = get_feature_names();
    features.variant = variant;
    features.extraction_date = datestr(now);
    
    % Save features
    save(output_file, 'features');
    
    fprintf('Feature extraction completed!\n');
    fprintf('Total samples: %d\n', size(all_features, 1));
    fprintf('Feature dimension: %d\n', size(all_features, 2));
    fprintf('Features saved to: %s\n', output_file);
end

function img_features = extract_single_image_features(image_path)
    % Extract features from a single image
    %
    % Args:
    %   image_path: Path to input image
    %
    % Returns:
    %   img_features: Feature vector for the image
    
    % Read and preprocess image
    img = imread(image_path);
    
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    
    % Resize to standard size
    img = imresize(img, [224, 224]);
    gray = imresize(gray, [224, 224]);
    
    % Initialize feature vector
    img_features = [];
    
    %% 1. Color features (if RGB image)
    if size(img, 3) == 3
        % Color histograms
        hist_r = imhist(img(:,:,1), 32)';
        hist_g = imhist(img(:,:,2), 32)';
        hist_b = imhist(img(:,:,3), 32)';
        
        % Color moments
        mean_r = mean(img(:,:,1), 'all');
        mean_g = mean(img(:,:,2), 'all');
        mean_b = mean(img(:,:,3), 'all');
        
        std_r = std(double(img(:,:,1)), 0, 'all');
        std_g = std(double(img(:,:,2)), 0, 'all');
        std_b = std(double(img(:,:,3)), 0, 'all');
        
        color_features = [hist_r, hist_g, hist_b, mean_r, mean_g, mean_b, std_r, std_g, std_b];
        img_features = [img_features, color_features];
    end
    
    %% 2. Texture features
    % GLCM features
    glcm = graycomatrix(gray, 'Offset', [0 1; -1 1; -1 0; -1 -1], 'Symmetric', true);
    glcm_stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    texture_features = [mean(glcm_stats.Contrast), mean(glcm_stats.Correlation), ...
                       mean(glcm_stats.Energy), mean(glcm_stats.Homogeneity)];
    
    % Local Binary Pattern (simplified)
    lbp_features = extract_lbp_features(gray);
    
    img_features = [img_features, texture_features, lbp_features];
    
    %% 3. Shape features (using edge detection)
    edges = edge(gray, 'canny');
    
    % Edge density
    edge_density = sum(edges(:)) / numel(edges);
    
    % Edge orientation histogram
    [Gx, Gy] = gradient(double(gray));
    edge_angles = atan2(Gy, Gx);
    edge_hist = histcounts(edge_angles(edges), 8);
    edge_hist = edge_hist / sum(edge_hist);  % Normalize
    
    shape_features = [edge_density, edge_hist];
    img_features = [img_features, shape_features];
    
    %% 4. Statistical features
    % Intensity statistics
    mean_intensity = mean(gray, 'all');
    std_intensity = std(double(gray), 0, 'all');
    skewness_intensity = skewness(double(gray(:)));
    kurtosis_intensity = kurtosis(double(gray(:)));
    
    stat_features = [mean_intensity, std_intensity, skewness_intensity, kurtosis_intensity];
    img_features = [img_features, stat_features];
end

function lbp_features = extract_lbp_features(gray_img)
    % Extract simplified Local Binary Pattern features
    %
    % Args:
    %   gray_img: Grayscale image
    %
    % Returns:
    %   lbp_features: LBP histogram features
    
    [rows, cols] = size(gray_img);
    lbp = zeros(rows-2, cols-2);
    
    % Simplified 3x3 LBP
    for i = 2:rows-1
        for j = 2:cols-1
            center = gray_img(i, j);
            
            % 8 neighbors
            neighbors = [gray_img(i-1,j-1), gray_img(i-1,j), gray_img(i-1,j+1), ...
                        gray_img(i,j+1), gray_img(i+1,j+1), gray_img(i+1,j), ...
                        gray_img(i+1,j-1), gray_img(i,j-1)];
            
            % Binary pattern
            binary_pattern = neighbors >= center;
            
            % Convert to decimal
            lbp_value = sum(binary_pattern .* [1, 2, 4, 8, 16, 32, 64, 128]);
            lbp(i-1, j-1) = lbp_value;
        end
    end
    
    % Create histogram (uniform patterns only)
    lbp_hist = histcounts(lbp(:), 0:255);
    lbp_features = lbp_hist / sum(lbp_hist);  % Normalize
    
    % Keep only first 32 bins for dimensionality reduction
    lbp_features = lbp_features(1:32);
end

function feature_names = get_feature_names()
    % Get names of extracted features
    %
    % Returns:
    %   feature_names: Cell array of feature names
    
    feature_names = {};
    
    % Color histogram features (RGB)
    for i = 1:32
        feature_names{end+1} = sprintf('hist_r_%d', i);
    end
    for i = 1:32
        feature_names{end+1} = sprintf('hist_g_%d', i);
    end
    for i = 1:32
        feature_names{end+1} = sprintf('hist_b_%d', i);
    end
    
    % Color moments
    feature_names = [feature_names, {'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b'}];
    
    % Texture features
    feature_names = [feature_names, {'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity'}];
    
    % LBP features
    for i = 1:32
        feature_names{end+1} = sprintf('lbp_%d', i);
    end
    
    % Shape features
    feature_names{end+1} = 'edge_density';
    for i = 1:8
        feature_names{end+1} = sprintf('edge_orient_%d', i);
    end
    
    % Statistical features
    feature_names = [feature_names, {'mean_intensity', 'std_intensity', 'skewness_intensity', 'kurtosis_intensity'}];
end
