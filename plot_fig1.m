%% plot_fig1.m
% Reproduce Figure 1 (iflex_dispatch_pattern) in MATLAB from the exported data.
% Data is produced by export_fig1_data.py ->
%   data/iflex_dispatch_pattern_data.mat
%
% Style matches the case-study figures: Helvetica (MATLAB default), FontSize 16,
% boxed axes with a light grid. Household response vs. cumulative recent event
% exposure, by incentive-price quartile plus the pooled curve.
clc; clear; close all;

S = load('data/iflex_dispatch_pattern_data.mat');

names  = cellstr(S.names);          % G x 1 cell of labels
colors = S.colors;                  % G x 3 RGB in [0,1]
G      = numel(names);
FS     = 20;

figure('Position',[100 100 700 350]);
ax = axes; hold(ax,'on'); box(ax,'on');

% zero reference line
yline(ax, 0, ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1.4, 'HandleVisibility','off');

h = gobjects(G,1);
for g = 1:G
    x   = S.x(g,:);   y = S.y(g,:);   e = S.err(g,:);
    m   = ~isnan(x);                              % drop NaN padding
    x   = x(m);  y = y(m);  e = e(m);
    isPool = strcmp(names{g}, 'Pool (Q1-Q4)');
    if isPool
        lw = 1.9;  ms = 6;
    else
        lw = 1.9;  ms = 6;
    end
    h(g) = errorbar(ax, x, y, e, 'o-', ...
        'Color', colors(g,:), 'MarkerFaceColor', colors(g,:), ...
        'LineWidth', lw, 'MarkerSize', ms, 'CapSize', 3);
end

xlabel(ax, 'Rolling 168-hour sum of past response (kW)', 'FontSize', FS);
ylabel(ax, 'Demand reduction (kW)', 'FontSize', FS);
ylim([-0.4, 0.7])

grid(ax,'on');
set(ax, 'FontSize', FS, 'LineWidth', 0.9, 'Layer','top', ...
        'TickDir','in', 'GridColor',[0.75 0.75 0.75], 'GridAlpha',0.5, ...
        'XColor',[0.15 0.15 0.15], 'YColor',[0.15 0.15 0.15]);

% legend
lgd = legend(ax, h, names, 'Location','southeast', 'Box','off', 'FontSize',16);
title(lgd, S.legend_title);

% OLS-slope annotation box (top-left)
lines = {'OLS slope:'};
for g = 1:G
    tag = names{g};
    tok = regexp(tag, '^(Q\d|Pool)', 'match', 'once');
    if isempty(tok), tok = tag; end
    lines{end+1} = sprintf('  %-5s %+.4f  (SE %.4f)', [tok ':'], S.slope(g), S.se(g)); %#ok<SAGROW>
end
text(ax, 0.02, 0.999, lines, 'Units','normalized', ...
     'VerticalAlignment','top', 'HorizontalAlignment','left', ...
     'FontSize', 16, ...
     'Margin', 4);

% export to vector PDF (matches the LaTeX include)
outPDF = 'iflex_dispatch_pattern.pdf';
try
    exportgraphics(gcf, outPDF, 'ContentType','vector');
catch
    set(gcf,'PaperPositionMode','auto');
    print(gcf, outPDF, '-dpdf', '-vector');
end
% fprintf('Saved %s\n', outPDF);
