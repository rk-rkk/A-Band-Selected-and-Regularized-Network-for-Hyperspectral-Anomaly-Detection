clear
clc
close all

file_name = 'AVIRIS1.mat';
map_name = strcat('../data/',file_name);
load(map_name);
[w,h,~] = size(data);
w_new = uint32(w)/3*3;
h_new = uint32(h)/3*3;

RXD_name = 'AVIRIS1.mat';
RXD_name = strcat('../data/RXD_',RXD_name);
load(RXD_name);
map_all(:,1) = detection';

file_name_FrFT = 'AVIRIS1_0_8.mat';
FrFT_name = strcat('../data/',file_name_FrFT);
load(FrFT_name);
map_all(:,2) = r2';


FEB_name = strcat('../data/FEBPAD-',file_name);
load(FEB_name);
temp = reshape(y,w_new*h_new,1);
map_all(:,3) = temp;


file_name_auto = 'AVIRIS-1_detection.mat';
Auto_name = strcat('../data/AutoAD_',file_name_auto);
load(Auto_name);
y = double(detection(1:w_new,1:h_new));
temp = reshape(y,w_new*h_new,1);
map_all(:,4) = temp;


file_name_RGAE = 'AVIRIS-1-0818.mat';
RGAE_name = strcat('../data/RGAE_',file_name_RGAE);
load(RGAE_name);
y = double(y(1:w_new,1:h_new));
temp = reshape(y,w_new*h_new,1);
map_all(:,5) = temp;


file_name_PDBS = 'PDBS_AVIRIS1';
PDBS_name = strcat('../data/',file_name_PDBS);
load(PDBS_name);
temp = reshape(double(detectmap),w_new*h_new,1);
map_all(:,6) = temp;

file_name_BSRegNet = 'BSRegNet-AVIRIS1';
BSReg_name = strcat('../data/',file_name_BSRegNet);
load(BSReg_name);
temp = reshape(double(detectmap),w_new*h_new,1);
map_all(:,7) = temp;

detec_label(1) = {'RXD'};
detec_label(2) = {'FrFT'};
detec_label(3) = {'FEBPAD'};
detec_label(4) = {'AutoAD'};
detec_label(5) = {'RGAE'};
detec_label(6) = {'PDBS'};
detec_label(7) = {'BSRegNet'};

map = map(1:w_new,1:h_new);
GT = reshape(map,w_new*h_new,1);

det_map = map_all;

num_map = size(det_map, 2);
for i = 1:num_map
    det_map(:, i) = (det_map(:, i) - min(det_map(:, i))) / (max(det_map(:, i)) - min(det_map(:, i)));
end

% PD and PF based on uniform step and sample value
for k = 1:num_map
    tau1(:, k) = (0:0.01:1)';
end
tau1 = sort(tau1, 'descend');

tau2 = sort(det_map, 'descend');

for k = 1:num_map
    for i = 1:length(tau2)
        map = det_map(:, k);

        map(det_map(:, k) >= tau2(i, k)) = 1;
        map(det_map(:, k) < tau2(i, k)) = 0;

        [PD2(i, k), PF2(i, k)] = cal_pdpf(map, GT);
    end
end

figure, 
plot(PF2, PD2, 'LineWidth', 2)
axis([0, 1, 0, 1])

set(gca, 'XTick', 0:0.2:1, 'fontsize', 16)
set(gca, 'YTick', 0:0.2:1, 'fontsize', 16)
xlabel('PF', 'fontsize', 18)
ylabel('PD', 'fontsize', 18)
grid on

for i = 1:num_map
%     name1(i) = strcat(detec_label(i), ',', '{\Delta}', '=0.01');
    name2(i) = detec_label(i);
end
% legend([name1,name2])
legend(name2)
legend boxoff
