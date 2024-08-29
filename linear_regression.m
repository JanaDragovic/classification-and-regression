% Napomena: Kako bi carbig skup podataka bio dostupan potrebno je
% instalirati 'Statistics and Machine Learning Toolbox' za Desktop verziju
% MATLAB-a ili jednostavno pokrenuti dati kod u Online MATLAB okruzenju

% U slucaju izrade ovog projekta, kod je pokrenut u Online okruzenju

% Ucitavanje podataka
load carbig

% Priprema podataka
X = [Displacement, Horsepower, Weight, Acceleration]; 
Y = MPG; 

% Uklanjanje NaN vrednosti
validData = ~isnan(Y) & all(~isnan(X), 2);
X = X(validData, :);
Y = Y(validData);

% Podela na trening i test skupove (70% trening, 30% test)
cv = cvpartition(length(Y), 'Holdout', 0.3);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest = X(test(cv), :);
YTest = Y(test(cv));

% Izrada modela sa jednim prediktorom
mdl1 = fitlm(XTrain(:,1), YTrain); % Model za Displacement
mdl2 = fitlm(XTrain(:,2), YTrain); % Model za Horsepower
mdl3 = fitlm(XTrain(:,3), YTrain); % Model za Weight
mdl4 = fitlm(XTrain(:,4), YTrain); % Model za Acceleration

% Izrada modela sa svim prediktorima
mdlAll = fitlm(XTrain, YTrain);

% Evaluacija modela sa jednim prediktorom
Ypred1 = predict(mdl1, XTest(:,1));
Ypred2 = predict(mdl2, XTest(:,2));
Ypred3 = predict(mdl3, XTest(:,3));
Ypred4 = predict(mdl4, XTest(:,4));

MSE1 = mean((YTest - Ypred1).^2);
MSE2 = mean((YTest - Ypred2).^2);
MSE3 = mean((YTest - Ypred3).^2);
MSE4 = mean((YTest - Ypred4).^2);

% Evaluacija modela sa svim prediktorima
YpredAll = predict(mdlAll, XTest);
MSEAll = mean((YTest - YpredAll).^2);

fprintf('MSE za Displacement: %.2f\n', MSE1);
fprintf('MSE za Horsepower: %.2f\n', MSE2);
fprintf('MSE za Weight: %.2f\n', MSE3);
fprintf('MSE za Acceleration: %.2f\n', MSE4);
fprintf('MSE sa svim prediktorima: %.2f\n', MSEAll);

% Scatter plot sa regresionom linijom za svaki prediktor
figure;
subplot(2,2,1);
plot(XTest(:,1), YTest, 'o');
hold on;
plot(XTest(:,1), Ypred1, '-r');
title('Displacement vs MPG');
xlabel('Displacement');
ylabel('MPG');

subplot(2,2,2);
plot(XTest(:,2), YTest, 'o');
hold on;
plot(XTest(:,2), Ypred2, '-r');
title('Horsepower vs MPG');
xlabel('Horsepower');
ylabel('MPG');

subplot(2,2,3);
plot(XTest(:,3), YTest, 'o');
hold on;
plot(XTest(:,3), Ypred3, '-r');
title('Weight vs MPG');
xlabel('Weight');
ylabel('MPG');

subplot(2,2,4);
plot(XTest(:,4), YTest, 'o');
hold on;
plot(XTest(:,4), Ypred4, '-r');
title('Acceleration vs MPG');
xlabel('Acceleration');
ylabel('MPG');

% Histogram reziduala za svaki model
figure;
subplot(2,2,1);
histogram(YTest - Ypred1, 10, 'FaceColor', 'b', 'EdgeColor', 'k');
title('Reziduali za Displacement');
xlabel('Rezidual');
ylabel('Frekvencija');
ylim([0 40]); % Podesavanje y-ose
grid on;

subplot(2,2,2);
histogram(YTest - Ypred2, 10, 'FaceColor', 'g', 'EdgeColor', 'k');
title('Reziduali za Horsepower');
xlabel('Rezidual');
ylabel('Frekvencija');
ylim([0 40]); % Podesavanje y-ose
grid on;

subplot(2,2,3);
histogram(YTest - Ypred3, 10, 'FaceColor', 'm', 'EdgeColor', 'k');
title('Reziduali za Weight');
xlabel('Rezidual');
ylabel('Frekvencija');
ylim([0 40]); % Podesavanje y-ose
grid on;

subplot(2,2,4);
histogram(YTest - Ypred4, 10, 'FaceColor', 'c', 'EdgeColor', 'k');
title('Reziduali za Acceleration');
xlabel('Rezidual');
ylabel('Frekvencija');
ylim([0 40]); % Podesavanje y-ose
grid on;


% Grafikon važnosti prediktora (koeficijenata)
figure;
bar(abs(mdlAll.Coefficients.Estimate(2:end)));
title('Važnost prediktora');
xlabel('Prediktor');
ylabel('Apsolutna vrednost koeficijenta');
xticklabels({'Displacement', 'Horsepower', 'Weight', 'Acceleration'});
