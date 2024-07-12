using DelimitedFiles;

IsDir = "data/Is/";

I1data = readdlm(string(IsDir, "I1_1.dat"));
I2data = readdlm(string(IsDir, "I2_1.dat"));
writedlm(string(IsDir, "I1_1.csv"), I1data, ',');
writedlm(string(IsDir, "I2_1.csv"), I2data, ',');

for i=101:50:501
    I1data = readdlm(string(IsDir, "I1_", i, ".dat"));
    I2data = readdlm(string(IsDir, "I2_", i, ".dat"));
    writedlm(string(IsDir, "I1_", i, ".csv"), I1data, ',');
    writedlm(string(IsDir, "I2_", i, ".csv"), I2data, ',');
end
    
