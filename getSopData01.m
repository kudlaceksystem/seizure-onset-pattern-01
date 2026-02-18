close all; clear
snlp = 'r:\Kudlacek\Seizure onset patterns\Data for Premek\Seizures 5000 Hz 260102\All\';
lblp = 'r:\Kudlacek\Seizure onset patterns\Data for Premek\Seizures 5000 Hz 260102\All\Label JK sz ons\';
par.lblClassName = "Seizure"; % Signal extraction parameters
par.onsetDurationS = 5;
par.marginS = 10; % Margin for plotting around the onset
par.fs = 5000;
par.preprocessFilterF = [1 100]; % Frequencies of the preprocess filter
par.feaNames = ["15-70 Hz", "5-13 Hz", "LL", "AcorrMax", "SpectEnt"];

%% Get file names
d = dir([snlp, '*.mat']);
snln = {d.name}';
d = dir([lblp, '*-lbl3.mat']);
lbln = {d.name}';

numFiles = numel(lbln);
fea = NaN(numFiles, numel(par.feaNames)); % Feature vectors
sShort = NaN(numFiles, par.onsetDurationS*par.fs);
sLong = NaN(numFiles, par.onsetDurationS*par.fs + 2*par.marginS*par.fs);
for kf = 1 : numFiles
    disp(['Processing file ', num2str(kf), '/', num2str(numFiles)])
    [sShort(kf, :), sLong(kf, :), taxShort, taxLong] = loadSeizureOnset(lblp, lbln{kf}, snlp, snln{kf}, par);

    [fea(kf, 1), fea(kf, 2)] = bandpower_bands(sShort(kf, :), par.fs);
    fea(kf, 3) = sum(abs(diff(sShort(kf, :))));
    fea(kf, 4) = periodicityIndex(sShort(kf, :), par.fs);
    fea(kf, 5) = spectralEntropy(sShort(kf, :), par.fs);

    % figure('Position', [500 500 1500 300], 'Color', 'w');
    % 
    % axes('Position', [0.04 0.1 0.7 0.8])
    % plot(taxLong, sLong(kf, :))
    % hold on
    % plot(taxShort, sShort(kf, :))
    % addFeatureText(fea(kf, :), feaNames)
    % title(snln{kf})
    % hold off
    % 
    % haxBarh = axes('Position', [0.87 0.1 0.12 0.8]);
    % scaledFea = fea(kf, :).*[10 10 0.1 10 1];
    % barh(haxBarh, scaledFea)
    % haxBarh.YDir = 'reverse';
    % haxBarh.YTickLabel = feaNames;
    % 
    % drawnow
    % savefig(gcf, ['./fig/', snln{kf}(1:end-4), '.fig'])
end
disp('Saving data')
save("sopData.mat", "par", "snln", "fea", "sShort", "sLong", "taxShort", "taxLong")
disp('Finished')








function [sShort, sLong, taxShort, taxLong] = loadSeizureOnset(lblp, lbln, snlp, snln, par)
    load(fullfile(lblp, lbln), 'sigInfo', 'lblSet')
    load(fullfile(snlp, snln), 'sigTbl')
    nch = find(startsWith(sigTbl.ChName, "L-")); % Number of the channel to process
    if sigInfo.SigStart(nch) ~= sigTbl.SigStart(nch)
        error('_jk Different SigStart in sigInfo and sigTbl.')
    end
    lblSetToUse = lblSet(lblSet.ClassName == par.lblClassName, :);
    lblSetToUse = lblSetToUse(1, :);
    fsOrig = sigTbl.Fs(nch);
    s = fillmissing(sigTbl.Data{nch}, 'linear');
    [p, q] = rat(par.fs/fsOrig, 1e-12);  % rational approx with tight tolerance
    s = resample(s, p, q);
    s = flt(s, par.fs, par.preprocessFilterF);
    st = fix(seconds(lblSetToUse.Start - sigInfo.SigStart(nch))*par.fs);
    en = st + fix(par.onsetDurationS*par.fs) - 1;
    sShort = s(st : en);
    taxShort = 1/par.fs : 1/par.fs : numel(sShort)/par.fs;
    sLong = s(st - par.marginS*par.fs : en + par.marginS*par.fs);
    taxLong = (1/par.fs : 1/par.fs : numel(sLong)/par.fs) - 10;
end
function filteredSignal = flt(s, fs, preprocessFilterF)
    [b, a] = butter(2, preprocessFilterF/(fs/2));
    filteredSignal = filtfilt(b, a, s);
end
function [FAI, SRAI, P_15_70, P_1_70, f, Pxx] = bandpower_bands(s, fs)
    % bandpower_bands  Compute spectral power in 15-70 Hz and 1-70 Hz bands.
    %   FAI ... fast activity index
    %   SRAI .. shart rhythmic activity index
    %   [P_15_70, P_1_70, f, Pxx] = bandpower_bands(s, fs)
    %   s   : input signal (vector, double)
    %   fs  : sample rate (Hz)
    %   P_15_70 : power in 15-70 Hz (units^2)
    %   P_1_70  : power in 1-70 Hz  (units^2)
    %   f   : frequency vector corresponding to Pxx
    %   Pxx : power spectral density estimate (units^2/Hz)
    %
    % Uses Welch's method and numerical integration of the PSD.
    
    % Ensure column vector
    s = s(:);
    
    % Parameters for pwelch: 2-second Hamming window (or shorter if signal short)
    winSec = min(2, max(1, numel(s)/fs));   % between 1 and 2 s
    win = round(winSec * fs);
    noverlap = round(0.5 * win);
    nfft = max(2^nextpow2(win), 1024);
    
    % PSD estimate (one-sided)
    [Pxx, f] = pwelch(s, win, noverlap, nfft, fs);
    
    % Helper to integrate PSD across band [f1 f2]
    integrateBand = @(f1,f2) trapz(f(f>=f1 & f<=f2), Pxx(f>=f1 & f<=f2));
    
    
    P_1_70  = integrateBand(1, 70);
    P_15_70 = integrateBand(15, 70);
    P_5_13 = integrateBand(5, 13);
    FAI = P_15_70/P_1_70;
    SRAI = P_5_13/P_1_70;
end
function pi_idx = periodicityIndex(s, fs)
    % periodicityIndex  Max normalized autocorrelation peak for lags 0.25–2 s.
    %   pi_idx = periodicityIndex(s, fs)
    %   Captures repetition in 0.5–4 Hz (lags 0.25–2 s).
    %   Uses unbiased/autocov normalization via xcorr('coeff').
    
    s = s(:);
    N = numel(s);
    
    % Remove mean to focus on periodicity
    s = s - mean(s);
    
    % Compute normalized autocorrelation (coeff)
    [acf, lags] = xcorr(s, s, round(4*fs), 'coeff'); % up to 2 s lag
    % lags in samples, center at zero
    lagSec = lags / fs;
    
    % Consider positive lags only, excluding zero lag
    posMask = (lagSec >= 0.25) & (lagSec <= 2.0);
    acfPos = acf(posMask);
    % % % % % % % % plot(lagSec, acf)
    % % % % % % % % hold on
    % % % % % % % % plot(lagSec(posMask), acfPos)
    % % % % % % % % hold off
    % % % % % % % % pause
    % Maximum peak in the window
    pi_idx = max(acfPos);
end
function sent = spectralEntropy(s, fs)
    % spectralEntropy  Shannon entropy of normalized PSD in 1–70 Hz.
    %   sent = spectralEntropy(s, fs)
    %   Returns entropy in bits.
    
    s = s(:);
    
    % PSD via Welch
    win = round(1 * fs);
    noverlap = round(0.5 * win);
    nfft = max(2^nextpow2(win), 2048);
    [Pxx, f] = pwelch(s, win, noverlap, nfft, fs);
    
    % Restrict to 1-70 Hz
    mask = (f >= 1) & (f <= 70);
    p = Pxx(mask);
    
    % Normalize to a probability distribution
    p = p(:);
    p_sum = sum(p);
    if p_sum <= 0
        sent = 0;
        return
    end
    p = p / p_sum;
    
    % Shannon entropy (bits), avoid log(0)
    p(p <= eps) = []; % drop zero entries
    sent = -sum(p .* log2(p));
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

