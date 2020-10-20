function template_nf(jsonfolder, filename, savefigloc)

input_params_filename = strcat(filename, '.json');
input_params_dir  = jsonfolder;
json_mode = 'read';
%% Load configuration file with options
input_params = read_write_json(input_params_filename, input_params_dir, json_mode);

%% Check that the input tmp folder and output folder exist and are consistent with OS,
% if they aren't, it will try to fix the problem, or error
input_params = check_storage_dirs(input_params, 'temp');
input_params = check_storage_dirs(input_params, 'output');

%% Run core functions: interpolation, estimation and classification, streamlines, this function writes to a new json file
output_params = main_core(input_params);

%% Run basic analysis
main_analysis(output_params);

%% Visualisation
speeds_fig_handle = plot1d_speed_distribution(output_params);
savefig(speeds_fig_handle, strcat(savefigloc, '\\', filename, '_Speeds.fig'))
close('all')
flows_fig_handle = plot2d_svd_modes(output_params);
savefig(flows_fig_handle, strcat(savefigloc, '\\', filename, '_Flows.fig'))
close('all')
%main_visualisation(output_params);

end 
