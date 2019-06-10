#!/usr/bin/octave -qf

pkg load statistics;

printf("\n\n\n\n")
arg_list = argv ();
dataset_name = arg_list{1};


# relative_path = "../"
# X_filename  = strcat(relative_path, '.datasets/',dataset_name, '/', dataset_name, "_prep_encoding2.csv")
# ctp_filename = strcat(relative_path, 'datasets/',dataset_name, '/', dataset_name, "_prep_encoding2_proj_ctp.csv")
# labels_filename  = strcat(relative_path, 'datasets/',dataset_name, '/', dataset_name, "_labels.csv")

relative_path = "../../"
X_filename  = strcat(relative_path, 'datasets/',dataset_name, '/', dataset_name, "_prep_encoding2.csv")
ctp_filename = strcat(relative_path, 'datasets/',dataset_name, '/', dataset_name, "_prep_encoding2_proj_ctp.csv")
labels_filename  = strcat(relative_path, 'datasets/',dataset_name, '/', dataset_name, "_labels.csv")


X              = dlmread(X_filename, ',', 0,0);
control_points = dlmread(ctp_filename, ',', 0,0);
labels         = dlmread(labels_filename, ',', 0,0);

n = size(X,1);
D = pdist2(X, X, 'euclidean');
options.k = 10;
options.nc = ceil(sqrt(n));
options.proj_type = 'force';
# options.debug = true;

[proj_points, proj_cp, sc] = LSP(D, control_points, options);

dlmwrite (strcat(relative_path, 'datasets/',dataset_name, '/', dataset_name, "_projected_octave.csv"), proj_points, ",")


# ctp_idx = control_points(:,3) +1;
# scatter(proj_points(:,1), proj_points(:,2),5, labels', "filled");
# hold on
# scatter(proj_cp(:,1), proj_cp(:,2),10, labels(ctp_idx), "filled");
# # scatter(control_points(:,1), control_points(:,2),10, labels(ctp_idx), "filled");
# pause(5)



