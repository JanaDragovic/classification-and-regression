% Napomena: Kako bi carbig skup podataka bio dostupan potrebno je
% instalirati 'Statistics and Machine Learning Toolbox' za Desktop verziju
% MATLAB-a ili jednostavno pokrenuti dati kod u Online MATLAB okruzenju

% U slucaju izrade ovog projekta, kod je pokrenut u Online okruzenju

% Ucitavanje podataka
load carbig

% Priprema podataka: koristimo karakteristike Displacement, Horsepower, Weight i Acceleration
X = [Displacement, Horsepower, Weight, Acceleration]; % Karakteristike (feature vectors)

% Prvo konvertovanje 'Origin' u cell array, pa u categorical
Y = categorical(cellstr(Origin)); % Oznake (labels) - Zemlja porekla

% Uklanjanje klasa koje nisu prisutne u trening skupu
validClasses = ismember(Y, categories(Y));
X = X(validClasses, :);
Y = Y(validClasses, :);

% Podela na trening i test skupove (70% trening, 30% test)
cv = cvpartition(Y, 'Holdout', 0.3); % Podela na trening i test skupove sa 30% podataka za testiranje
XTrain = X(training(cv), :); % Trening skup karakteristika
YTrain = Y(training(cv), :); % Trening skup oznaka
XTest = X(test(cv), :); % Test skup karakteristika
YTest = Y(test(cv), :); % Test skup oznaka

% Izrada LDA modela
ldaModel = fitcdiscr(XTrain, YTrain); % Treniranje LDA modela na trening skupu
ldaPredictions = predict(ldaModel, XTest); % Predikcija oznaka za test skup koristeci LDA model

% Izrada QDA modela
qdaModel = fitcdiscr(XTrain, YTrain, 'DiscrimType', 'quadratic'); % Treniranje QDA modela na trening skupu
qdaPredictions = predict(qdaModel, XTest); % Predikcija oznaka za test skup koristeci QDA model

% Izrada Naive Bayes modela
nbModel = fitcnb(XTrain, YTrain); % Treniranje Naive Bayes modela na trening skupu
nbPredictions = predict(nbModel, XTest); % Predikcija oznaka za test skup koristeci Naive Bayes model

% Prikaz matrice konfuzije za LDA model
figure;
confusionchart(YTest, ldaPredictions);
title('Matrica konfuzije za LDA');

% Prikaz matrice konfuzije za QDA model
figure;
confusionchart(YTest, qdaPredictions);
title('Matrica konfuzije za QDA');

% Prikaz matrice konfuzije za Naive Bayes model
figure;
confusionchart(YTest, nbPredictions);
title('Matrica konfuzije za Naive Bayes');

% Evaluacija modela
ldaAccuracy = sum(ldaPredictions == YTest) / length(YTest); % Racunanje tacnosti LDA modela
qdaAccuracy = sum(qdaPredictions == YTest) / length(YTest); % Racunanje tacnosti QDA modela
nbAccuracy = sum(nbPredictions == YTest) / length(YTest); % Racunanje tacnosti Naive Bayes modela

% Prikaz rezultata tacnosti
fprintf('LDA Tačnost: %.2f%%\n', ldaAccuracy * 100);
fprintf('QDA Tačnost: %.2f%%\n', qdaAccuracy * 100);
fprintf('Naive Bayes Tačnost: %.2f%%\n', nbAccuracy * 100);

% ROC kriva za LDA model
[~, scoresLDA] = predict(ldaModel, XTest);
[Xlda, Ylda, Tlda, AUCLDA] = perfcurve(YTest, scoresLDA(:, 2), 'USA');
figure;
plot(Xlda, Ylda);
xlabel('Stopa lažno pozitivnih');
ylabel('Stopa istinski pozitivnih');
title(['ROC Kriva za LDA (AUC = ' num2str(AUCLDA) ')']);

% ROC kriva za QDA model
[~, scoresQDA] = predict(qdaModel, XTest);
[Xqda, Yqda, Tqda, AUCQDA] = perfcurve(YTest, scoresQDA(:, 2), 'USA');
figure;
plot(Xqda, Yqda);
xlabel('Stopa lažno pozitivnih');
ylabel('Stopa istinski pozitivnih');
title(['ROC Kriva za QDA (AUC = ' num2str(AUCQDA) ')']);

% ROC kriva za Naive Bayes model
[~, scoresNB] = predict(nbModel, XTest);
[Xnb, Ynb, Tnb, AUCNB] = perfcurve(YTest, scoresNB(:, 2), 'USA');
figure;
plot(Xnb, Ynb);
xlabel('Stopa lažno pozitivnih');
ylabel('Stopa istinski pozitivnih');
title(['ROC Kriva za Naive Bayes (AUC = ' num2str(AUCNB) ')']);

% PCA smanjenje dimenzionalnosti
[coeff, score, ~] = pca(X);
figure;
gscatter(score(:,1), score(:,2), Y);
xlabel('Prva glavna komponenta');
ylabel('Druga glavna komponenta');
title('PCA Automobilskih Podataka');


