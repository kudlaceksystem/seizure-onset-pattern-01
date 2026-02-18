close all
clear

% numClus = 2;
% clusterColors = [1 0.9 0; 0 1 0.9];

% numClus = 3;
% clusterColors = [0.9 0.2 0; 0.1 0.8 0; 0 0.1 1];

numClus = 4;
clusterColors = [1 0 0.1; 1 0.6 0; 0 0.8 0.4; 0.2 0 1];


numSzPerFig = 8;
% figPos = [500 100 1200*21/30 1200]; % A4 paper
figPos = [500 100 1200*16/9 1200]; % 16:9 screen
axHeiMult = 0.7;
spiderColors = [0 0 1; 0 1 1; 0 1 0; 1 1 0; 1 0 0];

load("sopData.mat")

numSz = size(fea, 1);
numFea = size(fea, 2);
feaZsc = zscore(fea);

%% K-means
[idx,C,score,s] = clusterAndPlot(feaZsc, numClus, clusterColors);
print(['Seizure onsets clustering numClus', num2str(numClus, '%01d'), '.jpg'], '-djpeg', '-r600')




%% Plot individual seizures
feamin = min(fea, [], 1, 'omitnan');   % 1xN row vector of column minima
feamax = max(fea, [], 1, 'omitnan');   % 1xN row vector of column maxima
feaRsc = rescale(fea, 0, 1, 'InputMin', feamin, 'InputMax', feamax);
numFig = ceil(numSz/numSzPerFig);
numSzInEach = [numSzPerFig*ones(numFig-1, 1); rem(numSz-1, numSzPerFig)+1];
for kf = 1 : numFig
    close all
    hfMain = figure('Position', figPos, 'Color', 'w');
    for ks = 1 : numSzInEach(kf)
        axHei = axHeiMult*(1/numSzPerFig);
        axPosY = 1 - ks*(1/numSzPerFig) + (1-axHeiMult)/2*(1/numSzPerFig);
        haxSnl = axes(hfMain, 'Position', [0.03 axPosY 0.7 axHei]);
        plot(haxSnl, taxLong, sLong(ks+(kf-1)*numSzPerFig, :))
        hold on
        plot(haxSnl, taxShort, sShort(ks+(kf-1)*numSzPerFig, :))
        % % % addFeatureText(fea(kf, :), par.feaNames)
        title(snln{ks+(kf-1)*numSzPerFig}, "Interpreter", "none")
        haxSpd = axes(hfMain, 'Position', [0.8 axPosY 0.12 axHei]);
        spiderChartColored(feaRsc(ks+(kf-1)*numSzPerFig, :), par.feaNames, spiderColors)
    end
    print(['Seizure onsets ', num2str(kf, '%03d'), '.jpg'], '-djpeg', '-r600')
end






%% Functions
function [idx,C,score,s] = clusterAndPlot(Xz, k, colors)
% clusterAndPlot  K-means on z-scored data; show PC scatter and colored silhouette
%   [idx,C,score,s] = clusterAndPlot(Xz,k)
%   [idx,C,score,s] = clusterAndPlot(Xz,k,colors)
%
% Inputs
%   Xz     - n-by-p z-scored data (observations in rows)
%   k      - number of clusters (integer >=2)
%   colors - optional k-by-3 RGB matrix (values in [0,1]); default lines(k)
%
% Outputs
%   idx    - n-by-1 cluster indices (1..k)
%   C      - k-by-p centroids (in Xz space)
%   score  - n-by-p PCA scores
%   s      - n-by-1 silhouette values

if nargin < 2, error('Provide Xz and k.'); end
[n,p] = size(Xz);
if k < 2 || k > n, error('k must be between 2 and number of observations.'); end
if nargin < 3 || isempty(colors), colors = lines(k); else validateattributes(colors, {'numeric'},{'size',[k,3]}); end

% --- K-means ---
rng('default');
opts = statset('MaxIter',500);
[idx,C,~,~] = kmeans(Xz, k, 'Replicates', 10, 'Start', 'plus', 'Options', opts);

% --- PCA for plotting ---
[coeff, score, ~, ~, explained] = pca(Xz);

% Create one figure with two subplots: scatter (left) and silhouette (right)
hf = figure;
set(hf, 'Color', 'w');

% --- Left: scatter in PC1-PC2 ---
ax1 = subplot(1,2,1);
hold(ax1,'on');
scatter(ax1, score(:,1), score(:,2), 36, colors(idx,:), 'filled', 'MarkerEdgeColor',[0.2 0.2 0.2]);
Cpc = C * coeff(:,1:2); % project centroids
plot(ax1, Cpc(:,1), Cpc(:,2), 'kp', 'MarkerSize',12, 'MarkerFaceColor','y', 'LineWidth',1.2);
xlabel(ax1, sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(ax1, sprintf('PC2 (%.1f%%)', explained(2)));
title(ax1, sprintf('k-means (k=%d) in PC space', k));
grid(ax1,'on'); axis(ax1,'equal');
% legend
lh = gobjects(k,1);
for j = 1:k
    lh(j) = plot(ax1, NaN, NaN, 'o', 'MarkerFaceColor', colors(j,:), 'MarkerEdgeColor', [0.2 0.2 0.2]);
end
legend(ax1, lh, arrayfun(@(x) sprintf('Cluster %d',x), 1:k, 'UniformOutput', false), 'Location','bestoutside');
hold(ax1,'off');

% --- Right: compute silhouette values (no built-in plot) ---
% Using the function form that returns s without plotting:
s = silhouette(Xz, idx); % per documentation, when output requested it does not plot

% Build grouped bar-like silhouette manually so each group's bars get its color
ax2 = subplot(1,2,2);
hold(ax2,'on');

% Prepare grouping: for each cluster, sort its silhouette values descending
clusterOrder = 1:k; % we'll stack clusters in this order (1..k). You can reorder if desired.
yStart = 0;
yCenters = nan(n,1);
yClusterLimits = zeros(k,2); % [top bottom] for each cluster (for ticks)
plotHandles = gobjects(n,1);
idx_ordered = nan(n,1);
for cj = 1:k
    j = clusterOrder(cj);
    s_j = s(idx==j);
    nj = numel(s_j);
    if nj == 0
        yClusterLimits(j,:) = [NaN NaN];
        continue;
    end
    % sort within-cluster by silhouette descending (like MATLAB silhouette)
    [s_j_sorted, sortIdx] = sort(s_j, 'descend');
    % assign y positions for these bars: give each bar height 1 and small gap 0.1
    % We'll center bars at y = yStart + (nj:-1:1)
    ys = yStart + (nj:-1:1)';         % top-to-bottom numbering within this cluster
    % record for mapping back to original observations
    obsIdx = find(idx==j);
    obsIdx_sorted = obsIdx(sortIdx);  % indices in original data, in plotting order
    idx_ordered(yStart+1 : yStart+nj) = obsIdx_sorted; %#ok<AGROW>
    % draw rectangles for each bar
    barHeight = 0.9;
    for b = 1:nj
        yc = ys(b);
        yCenters(yStart + b) = yc;
        % Use rectangle to draw horizontal bar from 0 to s value
        x0 = 0;
        w = s_j_sorted(b);
        % ensure bars with negative silhouette extend left from zero like MATLAB
        if w >= 0
            xpos = x0;
            width = w;
        else
            xpos = w;
            width = -w;
        end
        r = rectangle(ax2, 'Position', [xpos, yc - barHeight/2, width, barHeight], ...
            'FaceColor', colors(j,:), 'EdgeColor', max(colors(j,:)*0.7, 0));
        plotHandles(yStart + b) = r;
    end
    % cluster vertical limits for ticks and separation lines
    topY = min(ys) + barHeight/2;
    bottomY = max(ys) - barHeight/2;
    yClusterLimits(j,:) = [topY, bottomY];
    % increment yStart
    yStart = yStart + nj;
    % add a small gap between clusters
    yStart = yStart + 1; % gap row
end

% Adjust axes to look like built-in silhouette
ylim(ax2, [0, max(yCenters)+1]);
xlim(ax2, [-1, 1]); % silhouette values are in [-1,1]
xlabel(ax2, 'Silhouette Value');
ylabel(ax2, 'Cluster');
title(ax2, 'Silhouette Plot (cluster-colored)');

% Draw horizontal separators and cluster labels
% Compute cluster centers for tick placement
clusterTickY = zeros(k,1);
for j = 1:k
    rows = find(idx==j);
    if isempty(rows)
        clusterTickY(j) = NaN;
    else
        % use mean of Y positions of that cluster's bars
        % Find the y positions we stored for those rows
        % mapping: we stored idx_ordered in plotting order; find all positions with idx==j
        posMask = idx_ordered(1:sum(~isnan(idx_ordered))) == j;
        ys_j = find(posMask);
        if ~isempty(ys_j)
            clusterTickY(j) = mean(ys_j);
        else
            clusterTickY(j) = NaN;
        end
    end
end
% clusterTickY are relative to positions; convert to the actual y coordinates used (we used ys values)
% Simpler: set ticks by using the midpoints of yClusterLimits
for j = 1:k
    if ~isnan(yClusterLimits(j,1))
        tickY(j) = mean(yClusterLimits(j,:));
    else
        tickY(j) = NaN;
    end
end
% Reverse y-direction so first cluster is at top (like builtin)
set(ax2, 'YDir','reverse');
% Place ticks at cluster midpoints (omit NaNs)
valid = ~isnan(tickY);
set(ax2, 'YTick', tickY(valid), 'YTickLabel', arrayfun(@(x) sprintf('Cluster %d',x), find(valid), 'UniformOutput', false));

% Add vertical line at zero
plot(ax2, [0 0], ylim(ax2), 'k:', 'LineWidth', 1);
grid(ax2,'on');
hold(ax2,'off');

end





% % % % function [idx,C,score,s] = clusterAndPlot(Xz, k, colors)
% % % % % clusterAndPlot  Run k-means on z-scored data, plot PCA scatter and silhouette
% % % % %   [idx,C,score,s] = clusterAndPlot(Xz,k)
% % % % %   [idx,C,score,s] = clusterAndPlot(Xz,k,colors)
% % % % %
% % % % % Inputs
% % % % %   Xz     - n-by-p data matrix (already z-scored; observations in rows)
% % % % %   k      - number of clusters (integer >=2)
% % % % %   colors - optional k-by-3 RGB matrix (values in [0,1]). If omitted uses lines(k).
% % % % %
% % % % % Outputs
% % % % %   idx    - n-by-1 cluster indices
% % % % %   C      - k-by-p centroids in Xz space
% % % % %   score  - n-by-p PCA scores (for plotting PC1/PC2)
% % % % %   s      - n-by-1 silhouette values
% % % % 
% % % % if nargin < 2
% % % %     error('Provide Xz and k.');
% % % % end
% % % % [n,p] = size(Xz);
% % % % if k < 2 || k > n
% % % %     error('k must be between 2 and number of observations.');
% % % % end
% % % % if nargin < 3 || isempty(colors)
% % % %     colors = lines(k);
% % % % else
% % % %     validateattributes(colors, {'numeric'}, {'size',[k,3]});
% % % % end
% % % % 
% % % % % K-means (robust defaults)
% % % % rng('default'); % reproducible; caller can change
% % % % opts = statset('MaxIter',500);
% % % % [idx,C,~,~] = kmeans(Xz, k, 'Replicates', 10, 'Start', 'plus', 'Options', opts);
% % % % 
% % % % % PCA for visualization
% % % % [coeff, score, ~, ~, explained] = pca(Xz);
% % % % 
% % % % % Scatter in PC1-PC2 with cluster colors
% % % % figure;
% % % % ax1 = gca;
% % % % hold(ax1,'on');
% % % % scatter(score(:,1), score(:,2), 36, colors(idx,:), 'filled', 'MarkerEdgeColor',[0.2 0.2 0.2]);
% % % % xlabel(ax1, sprintf('PC1 (%.1f%%)', explained(1)));
% % % % ylabel(ax1, sprintf('PC2 (%.1f%%)', explained(2)));
% % % % title(ax1, sprintf('k-means (k=%d) in PC space', k));
% % % % grid(ax1,'on');
% % % % axis(ax1,'equal');
% % % % 
% % % % % Draw cluster centroids in PC space (projected)
% % % % Cpc = C * coeff(:,1:2); % k-by-2
% % % % plot(ax1, Cpc(:,1), Cpc(:,2), 'kp', 'MarkerSize',12, 'MarkerFaceColor','y', 'LineWidth',1.2);
% % % % 
% % % % % Legend
% % % % % Create simple legend entries for clusters
% % % % lh = zeros(k,1);
% % % % for j = 1:k
% % % %     lh(j) = plot(ax1, NaN, NaN, 'o', 'MarkerFaceColor', colors(j,:), 'MarkerEdgeColor', [0.2 0.2 0.2]);
% % % % end
% % % % legend(lh, arrayfun(@(x) sprintf('Cluster %d',x), 1:k, 'UniformOutput', false), 'Location','bestoutside');
% % % % hold(ax1,'off');
% % % % 
% % % % % Silhouette: compute and plot. Use same distance as kmeans default (sqeuclidean).
% % % % figure;
% % % % % request silhouette to plot and return values and figure handle
% % % % [s, figH] = silhouette(Xz, idx); %#ok<ASGLU>
% % % % axSil = findobj(figH, 'Type', 'Axes');
% % % % if ~isempty(axSil)
% % % %     axSil = axSil(1);
% % % % else
% % % %     axSil = gca;
% % % % end
% % % % title(axSil, 'Silhouette Plot (colors matched to clusters)');
% % % % xlabel(axSil, 'Silhouette Value');
% % % % ylabel(axSil, 'Cluster');
% % % % 
% % % % % Find patch objects created by silhouette
% % % % patches = findobj(axSil, 'Type', 'Patch');
% % % % 
% % % % % If no patches found, just return
% % % % if isempty(patches)
% % % %     warning('No silhouette patch objects found; cannot recolor bars.');
% % % %     return;
% % % % end
% % % % 
% % % % % Determine mapping from patch objects to cluster indices.
% % % % % For each patch, use mean of its YData (vertical location).
% % % % patchMeans = nan(numel(patches),1);
% % % % for p = 1:numel(patches)
% % % %     y = patches(p).YData;
% % % %     y = y(~isnan(y));
% % % %     patchMeans(p) = mean(y);
% % % % end
% % % % [~, ordPatches] = sort(patchMeans, 'descend');   % top-to-bottom ordering of patches
% % % % 
% % % % % For clusters, compute mean of silhouette values and sort similarly
% % % % meanSi = zeros(k,1);
% % % % for j = 1:k
% % % %     meanSi(j) = mean(s(idx==j));
% % % % end
% % % % [~, ordClusters] = sort(meanSi, 'descend');
% % % % 
% % % % % Now align patches to clusters by ordering
% % % % patchesSorted = patches(ordPatches);
% % % % numToAssign = min(numel(patchesSorted), k);
% % % % for ii = 1:numToAssign
% % % %     pj = patchesSorted(ii);
% % % %     clusterIdx = ordClusters(ii);
% % % %     try
% % % %         pj.FaceColor = colors(clusterIdx, :);
% % % %         pj.EdgeColor = max(colors(clusterIdx,:)*0.7, 0);
% % % %     catch
% % % %         % older versions may not support direct property assignment; ignore silently
% % % %     end
% % % % end
% % % % 
% % % % % Return outputs
% % % % end

function spiderChartColored(values, names, colors)
% spiderChartColored  Radar chart with per-variable colors that blend from gray center.
%   spiderChartColored(values, names)
%   spiderChartColored(values, names, colors)
%
%   values - numeric vector (1xN or Nx1)
%   names  - cell array of char, cell array of string, or string array (N elements)
%   colors - optional N-by-3 RGB matrix (values in [0,1]); if omitted, uses distinct colors
%
% Example:
%   spiderChartColored([3 5 2 4 1], {'A','B','C','D','E'});
%   spiderChartColored([.2 .8 .6], ["x","y","z"], [1 0 0; 0 1 0; 0 0 1]);

% ---- Validate inputs ----
values = double(values(:))';
N = numel(values);
if N < 3
    error('Need at least 3 features.');
end

% names -> cellstr
if isstring(names) || iscellstr(names) || iscell(names)
    names = cellstr(names);
else
    error('names must be a cell array of char, cell array of string, or string array.');
end
if numel(names) ~= N
    error('Number of names (%d) must match values (%d).', numel(names), N);
end

% colors
if nargin < 3 || isempty(colors)
    % choose perceptually-distinct colors
    cmap = lines(max(N,3));        % lines is usually distinct
    colors = cmap(1:N,:);
else
    colors = double(colors);
    if ~isequal(size(colors), [N,3]) || any(colors(:) < 0) || any(colors(:) > 1)
        error('colors must be an N-by-3 RGB matrix with values in [0,1].');
    end
end

% ---- Scale values to radial [0,1] (preserve relative magnitudes) ----
maxV = max(values);
if maxV == 0
    valsR = zeros(size(values));
else
    valsR = values / maxV;
end

% geometry
theta = (0:N-1) * 2*pi / N;        % angles for outer vertices
thetaClosed = [theta theta(1)];
valsClosed = [valsR valsR(1)];

% radii scale (choose outer radius = 1)
rMax = 1;
outerR = valsR * rMax;

% compute coordinates of outer vertices (scaled by values)
xOuter = outerR .* cos(theta);
yOuter = outerR .* sin(theta);

% center is at (0,0)
% We will build a triangulated fan: center + vertex k + vertex k+1
% Build vertex list: first center, then outer vertices in order
vertices = [0 0; xOuter(:) yOuter(:)];   % (N+1) x 2
% Build faces (triangles) 1 = center, i+1 = vertex i, i+2 = vertex i+1
faces = zeros(N,3);
for k = 1:N
    v1 = 1;                 % center
    v2 = k+1;               % vertex k
    v3 = mod(k, N) + 2 - 1; % vertex k+1 -> careful: mod index
    % simpler:
    v3 = mod(k, N) + 1 + 1; % but ensure in range
    % correct v3:
    v3 = mod(k, N) + 1;     % gives 1..N -> add 1 for offset: +1 => 2..N+1
    faces(k,:) = [1, k+1, mod(k,N)+2-1]; % (we'll replace with robust formula below)
end
% Rebuild faces robustly:
faces = zeros(N,3);
for k = 1:N
    faces(k,1) = 1;                 % center
    faces(k,2) = k+1;               % vertex k
    faces(k,3) = mod(k,N) + 1 + 1;  % vertex k+1 (offset by 1 for center)
    % reduce: mod(k,N)+2 -> for k=N gives 2 (vertex 1), works
    faces(k,3) = mod(k,N) + 2 - 1;  % (this line harmless; we'll simplify below)
end
% Simpler correct faces:
faces = zeros(N,3);
for k = 1:N
    faces(k,:) = [1, k+1, mod(k, N)+1 + 1 - 1]; % dummy - will overwrite properly next
end
% Overwrite properly (clean implementation)
faces = zeros(N,3);
for k = 1:N
    faces(k,1) = 1;
    faces(k,2) = k+1;
    faces(k,3) = mod(k, N) + 2 - 1; % still confusing; do direct mapping:
end
% Final correct faces (clear previous confusion):
faces = zeros(N,3);
for k = 1:N
    faces(k,1) = 1;               % center
    faces(k,2) = k+1;             % kth outer vertex index in vertices
    if k < N
        faces(k,3) = k+2;         % next outer vertex
    else
        faces(k,3) = 2;           % wrap to first outer vertex
    end
end

% ---- Colors per vertex for interpolation ----
% center color = gray
centerColor = [0.6 0.6 0.6];   % adjustable
% vertex colors are the provided colors (pure colors)
% Build color array for each vertex in vertices order
vertexColors = zeros(N+1, 3);
vertexColors(1, :) = centerColor;
vertexColors(2:end, :) = colors;

% However: we want the color intensity to reflect the value: if value is small,
% the outer vertex position is near center, but color still pure at that vertex.
% With 'FaceColor','interp' interpolation across triangles will blend from centerColor
% at center to pure color at right vertex positions; since outer vertex moves with value,
% the visual effect is as requested.

% ---- Plot ----
hold on;
axis equal off;

% Draw grid (concentric polygons) using r ticks relative to rMax
numGrid = 4;
rTicks = linspace(0, rMax, numGrid+1);
for rt = rTicks
    xc = rt .* cos([theta theta(1)]);
    yc = rt .* sin([theta theta(1)]);
    plot(xc, yc, '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.9);
end

% radial spokes (full length rMax)
for k = 1:N
    plot([0 cos(theta(k))], [0 sin(theta(k))], '-', 'Color', [0.85 0.85 0.85]);
end

% Create patch using triangles and per-vertex color interpolation
hPatch = patch('Faces', faces, 'Vertices', vertices, ...
    'FaceVertexCData', vertexColors, 'FaceColor', 'interp', ...
    'EdgeColor', 'none', 'LineWidth', 1.2, 'FaceAlpha', 1.0);

% Draw outer polygon border and markers at outer vertices (use actual positions)
plot([xOuter xOuter(1)], [yOuter yOuter(1)], '-', 'Color', 0.5*[1 1 1], 'LineWidth', 0.3);
% plot(xOuter, yOuter, 'o', 'MarkerFaceColor', [0.1 0.4 0.6], 'MarkerEdgeColor', 'k');

% Labels
labelR = rMax * 1.12;
for k = 1:N
    th = theta(k);
    xt = labelR * cos(th);
    yt = labelR * sin(th);
    if abs(cos(th)) < 0.2
        ha = 'center';
    elseif cos(th) > 0
        ha = 'left';
    else
        ha = 'right';
    end
    if sin(th) > 0.2
        va = 'bottom';
    elseif sin(th) < -0.2
        va = 'top';
    else
        va = 'middle';
    end
    text(xt, yt, names{k}, 'HorizontalAlignment', ha, 'VerticalAlignment', va, 'FontWeight', 'normal');
end

% % radial tick labels on first axis
% for i = 2:numel(rTicks)
%     txt = num2str(rTicks(i) * maxV); % show original scale
%     text(rTicks(i) + 0.03*rMax, 0, txt, 'HorizontalAlignment','left', 'FontSize',8, 'Color',[0.2 0.2 0.2]);
% end

% limits and title
pad = rMax * 1.3;
xlim([-pad pad]);
ylim([-pad pad]);

hold off;
end
function th = addFeatureText(fea, feaNames, varargin)
    % addFeatureText  Add feature name:value text to current axes (or given axes).
    %   th = addFeatureText(fea, feaNames) places the five features in the top-left
    %   corner of the current axes (gca). fea is a numeric vector (length 5).
    %   feaNames is a cell array of 5 strings.
    %
    %   th = addFeatureText(..., 'Axes', ax)    target axes (default gca)
    %   th = addFeatureText(..., 'Position', pos)  normalized anchor [x y]
    %       pos is in axes normalized units (0..1), default = [0.02 0.98]
    %   th = addFeatureText(..., 'FontSize', fs)   default 10
    %   th = addFeatureText(..., 'Format', fmt)    value format, e.g. '%.3f'
    %   th = addFeatureText(..., Name,Value ...)   additional text properties
    %
    %   Returns array of text object handles.

    % Input parsing (lightweight)
    p = inputParser;
    addRequired(p,'fea',@(x)isnumeric(x) && isvector(x));
    addRequired(p,'feaNames',@(x)iscellstr(x) || isstring(x));
    addParameter(p,'Axes', [], @(x) isempty(x) || isgraphics(x,'axes'));
    addParameter(p,'Position',[0.02 0.98], @(x)isnumeric(x) && numel(x)==2);
    addParameter(p,'FontSize',10,@(x)isnumeric(x) && isscalar(x));
    addParameter(p,'Format','%.3f',@ischar);
    parse(p,fea,feaNames,varargin{:});
    ax = p.Results.Axes;
    if isempty(ax), ax = gca; end
    pos = p.Results.Position;
    fsz = p.Results.FontSize;
    fmt = p.Results.Format;

    fea = double(fea(:));        % ensure column
    n = numel(fea);
    names = cellstr(feaNames(:));

    % Safety: clip names/fea to available entries
    n = min(n,numel(names));
    fea = fea(1:n);
    names = names(1:n);

    % Compose lines "name: value"
    lines = cell(n,1);
    for k=1:n
        lines{k} = sprintf('%s: %s', names{k}, sprintf(fmt, fea(k)));
    end

    % Convert normalized axes position to data coordinates for text placement
    axUnitsOld = get(ax,'Units');
    set(ax,'Units','normalized');
    axPos = get(ax,'Position'); % not strictly needed, but keep intact
    set(ax,'Units',axUnitsOld);

    % Use axes normalized coordinates via annotation-like placement:
    % text supports 'Units','normalized' relative to axes when specifying Parent
    th = gobjects(n,1);
    holdState = ishold(ax);
    hold(ax,'on');

    % Vertical spacing in normalized units
    vpad = 0.08;                  % vertical step between lines
    x0 = pos(1);
    y0 = pos(2);

    for k=1:n
        % Each subsequent line shifts down
        yk = y0 - (k)*vpad;
        th(k) = text(ax, x0, yk, lines{k}, ...
            'Units','normalized', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','top', ...
            'FontSize', fsz,...
            'BackgroundColor','white');
    end

    if ~holdState
        hold(ax,'off');
    end
end
