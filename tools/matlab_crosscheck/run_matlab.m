function run_matlab(configPath)
%RUN_MATLAB  Reduce one real frame with PGIXS_matlab, headlessly.
%
%   run_matlab('tools/matlab_crosscheck/config.example.json')
%
% Replicates the reduction blocks of the PGIXS_matlab GUIs (which cannot be
% driven from a script) closely enough to compare their output against
% PGIXS_python's. Writes one CSV per reduction into the config's output_dir;
% run_python.py writes its own alongside and compare.py reads both.
%
% The GUI blocks reproduced here are:
%   GIXS 1D line cut   GIXS_gui.m  -> Anglecut1d_gi
%   GIXS azimuthal     GIXS_gui.m  -> DrawPlot_Callback (inlined below,
%                                     since the callback is nested in the GUI)
%   TrXS circular      TrXS_gui.m  -> CircularAve_Callback
% Keep them in sync when those callbacks change -- a silent drift here
% would make Python look wrong when it is not.

    cfg = jsondecode(fileread(configPath));
    cfg = applyDefaults(cfg);

    % PGIXS_matlab's own dist/ holds a gitignored packaged copy of
    % functions/; genpath would put it first and we would be measuring the
    % package, not the source. Call this after restoredefaultpath -- a stale
    % saved path survives across matlab -batch runs.
    repo = cfg.matlab_repo;
    allPaths = strsplit(genpath(repo), pathsep);
    allPaths = allPaths(~cellfun(@isempty, allPaths));
    addpath(strjoin(allPaths(~startsWith(allPaths, fullfile(repo, 'dist'))), pathsep));

    outDir = cfg.output_dir;
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    [~, ~, ext] = fileparts(cfg.frame);
    [rawData, ~] = loadPGIXSImageFile(cfg.frame, ext);

    param = struct();
    param.detector = cfg.detector;
    [param.Mask, param.pixel, ~] = loadPGIXSDetectorMask(param.detector);
    param.data = applyPGIXSMask(single(rawData), param.Mask);
    param.dataSize = size(param.data);
    param.bpos = cfg.beam_center(:).';
    param.SDD = cfg.sdd_mm;
    % Each side converts energy -> wavelength with its own h*c (MATLAB
    % 12.3985167 keV.A vs Python 12.3984198, a 7.8 ppm difference). Feeding
    % both the same *energy* rather than the same wavelength keeps that
    % visible in the compared q axes instead of hiding it, which is why the
    % conversion is duplicated here and in run_python.py.
    param.Energy = 12.398419843320026 / cfg.wavelength_a;

    fprintf('frame         : %s\n', cfg.frame);
    fprintf('data size     : %s\n', mat2str(param.dataSize));
    fprintf('pixel size    : %g mm\n', param.pixel);
    fprintf('energy        : %.10f keV\n', param.Energy);

    switch lower(cfg.mode)
        case 'gixs'
            param.ai = cfg.incident_angle_deg;
            param.QMapMet = 'Qxy/Qz';
            param = processGIXS2D(param, struct( ...
                'Energy', param.Energy, 'SDD', param.SDD, 'Beampos', param.bpos, ...
                'Detector', param.detector, 'IncAngle', param.ai, ...
                'QMapMethod', param.QMapMet));
        case 'trxs'
            param = processTrXS2D(param, struct( ...
                'Energy', param.Energy, 'SDD', param.SDD, 'Beampos', param.bpos, ...
                'Detector', param.detector));
        otherwise
            error('PGIXS:XCheck', 'mode must be "gixs" or "trxs", got "%s".', cfg.mode);
    end

    red = cfg.reductions;
    if isfield(red, 'line'),      runLineCuts(param, red.line, outDir);       end
    if isfield(red, 'azimuthal'), runAzimuthal(param, red.azimuthal, outDir); end
    if isfield(red, 'circular'),  runCircular(param, red.circular, outDir);   end
    fprintf('MATLAB DONE\n');
end

% ---------------------------------------------------------------- line cut
function runLineCuts(param, opts, outDir)
    param.Mapping = 'Pixel';
    param.yposmv = 0;
    param.zposmv = 0;
    for k = 1:numel(opts.angles_deg)
        ang = opts.angles_deg(k);
        param.ang = ang;
        [~, line1d] = Anglecut1d_gi(param);
        writeCsv(fullfile(outDir, sprintf('matlab_line%d.csv', ang)), ...
            'q,intensity', double(line1d(:, [2 3])));
        fprintf('line ang=%3d  : %d ray samples, q %.6f .. %.6f\n', ang, ...
            size(line1d, 1), min(line1d(:, 2)), max(line1d(:, 2)));
    end
end

% --------------------------------------------------------------- azimuthal
function runAzimuthal(param, opts, outDir)
    % Inlined from GIXS_gui.m's DrawPlot_Callback (a nested callback, so it
    % cannot be called directly). Polar sampling: one nearest pixel per
    % (theta, integer radius), averaged over radii.
    zOffset = 0;
    theta = (0:0.2:180)' + 180;
    radii = (opts.pixel + opts.range_minus) : (opts.pixel + opts.range_plus);

    [ThetaGrid, RGrid] = meshgrid(theta, radii);
    X_idx = round(RGrid .* cosd(ThetaGrid) + param.bpos(1));
    Y_idx = round(RGrid .* sind(ThetaGrid) + param.bpos(2) - zOffset);
    valid = (X_idx >= 1) & (X_idx <= param.dataSize(2)) & ...
            (Y_idx >= 1) & (Y_idx <= param.dataSize(1));

    D = param.data;
    if isfield(param, 'Mask') && isequal(size(param.Mask), size(D))
        gapAdj = dilatePGIXSGapMask(param.Mask, 3);
        idxTmp = sub2ind(size(D), Y_idx(valid), X_idx(valid));
        valid(valid) = ~gapAdj(idxTmp);
    end

    IndMap = nan(size(ThetaGrid), 'single');
    IndMap(valid) = sub2ind(size(D), Y_idx(valid), X_idx(valid));
    Vals = nan(size(IndMap), 'single');
    good = ~isnan(IndMap);
    Vals(good) = D(IndMap(good));
    Vals = Vals';

    if opts.use_median
        prof = median(Vals, 2, 'omitnan');
    else
        prof = mean(Vals, 2, 'omitnan');
    end
    counts = sum(~isnan(Vals), 2);
    prof(counts < 1) = NaN;

    writeCsv(fullfile(outDir, 'matlab_azimuthal.csv'), 'angle_deg,intensity,counts', ...
        [double(theta) - 180, double(prof), double(counts)]);
    fprintf('azimuthal     : %d angles at 0.2 deg, %d NaN, radii %s\n', ...
        numel(prof), sum(~isfinite(prof)), mat2str(radii));
end

% ------------------------------------------------------- circular average
function runCircular(param, opts, outDir)
    % Copied from TrXS_gui.m's CircularAve_Callback.
    I = single(param.data);
    Q = single(param.Q);
    invalid_geom = ~isfinite(Q) | (Q <= 0);
    if isfield(param, 'Mask') && isequal(size(param.Mask), size(I))
        I(dilatePGIXSGapMask(param.Mask, 3)) = NaN;
    end

    valid_pix = ~invalid_geom;
    isFullCircle = (opts.angle_min == opts.angle_max) || ...
        (opts.angle_min == 0 && opts.angle_max == 360);
    if ~isFullCircle
        valid_pix = valid_pix & buildSectorMask(size(I), param.bpos, ...
            opts.angle_min, opts.angle_max);
    end
    if ~any(valid_pix(:) & isfinite(I(:)))
        error('PGIXS:XCheck', 'No valid pixels after masking.');
    end

    radiusPrep = buildPGIXSRadiusBinMap(Q, [param.bpos(1), param.bpos(2)], valid_pix);
    [mu, ~, dataCounts] = processPGIXSRadiusBin1D(I, radiusPrep, opts.use_median, true);

    sectorCounts = double(radiusPrep.geom_counts(:));
    coverage = nan(size(mu), 'single');
    has = sectorCounts > 0;
    coverage(has) = single(double(dataCounts(has)) ./ sectorCounts(has));
    mu(coverage < opts.min_coverage) = NaN;

    writeCsv(fullfile(outDir, 'matlab_circular.csv'), 'q,intensity,coverage,r_px', ...
        [double(radiusPrep.q_axis(:)), double(mu(:)), double(coverage(:)), ...
         double(radiusPrep.r_axis(:))]);
    fprintf('circular      : %d bins, %d NaN, q %.6f .. %.6f\n', numel(mu), ...
        sum(~isfinite(mu)), min(radiusPrep.q_axis), max(radiusPrep.q_axis));
end

function m = buildSectorMask(sz, bpos, angMin, angMax)
    [cols, rows] = meshgrid(1:sz(2), 1:sz(1));
    ang = mod(atan2d(-(rows - bpos(2)), -(cols - bpos(1))), 360);
    m = ang >= angMin & ang <= angMax;
end

% -------------------------------------------------------------- utilities
function cfg = applyDefaults(cfg)
    if ~isfield(cfg, 'incident_angle_deg'), cfg.incident_angle_deg = 0.1; end
    if ~isfield(cfg, 'detector'), cfg.detector = 'Eiger4M'; end
    if ~isfield(cfg, 'reductions'), cfg.reductions = struct(); end

    if isfield(cfg.reductions, 'azimuthal')
        cfg.reductions.azimuthal = fillFields(cfg.reductions.azimuthal, ...
            {'range_minus', -1; 'range_plus', 1; 'use_median', false});
    end
    if isfield(cfg.reductions, 'circular')
        cfg.reductions.circular = fillFields(cfg.reductions.circular, ...
            {'angle_min', 0; 'angle_max', 360; 'min_coverage', 0.8; 'use_median', false});
    end
end

function s = fillFields(s, defaults)
    for ii = 1:size(defaults, 1)
        if ~isfield(s, defaults{ii, 1})
            s.(defaults{ii, 1}) = defaults{ii, 2};
        end
    end
end

function writeCsv(path, header, M)
    fid = fopen(path, 'w');
    if fid < 0, error('PGIXS:XCheck', 'Cannot write %s', path); end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# %s\n', header);
    fmt = [repmat('%.10g,', 1, size(M, 2) - 1), '%.10g\n'];
    fprintf(fid, fmt, M.');
end
