# %%

###############################################################################
# 1) Importy knihoven
###############################################################################
import numpy as np
import pandas as pd
import pyreadstat

# Pro jednoduché imputace
from sklearn.impute import SimpleImputer

# Pro regresní a stochastickou imputaci
from sklearn.linear_model import LinearRegression

# Pro vícenásobnou imputaci (IterativeImputer), který lze považovat 
# za jednu z metod MICE-like přístupů
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Pro rozpoznávání vzoru chybějících dat
# (Lze využít i specializované knihovny, např. "missingno" na vizualizaci.)
import missingno as msno

# Pro hodnocení modelů a imputací
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
from phik import phik_matrix

import scipy

from pyampute.exploration.mcar_statistical_tests import MCARTest
from scipy.stats import chi2


N = 1000
np.random.seed(6945)

result = pyreadr.read_r(".../easySHARE_rel9-0-0_R/easySHARE_rel9_0_0.rda") 


print(result.keys()) # let's check what objects we got

df = result['easySHARE_rel9_0_0'] # extract the pandas data frame for object df

###############################################################################
# Načtení datasetu + EDA
###############################################################################
# %%

df_9 = df.loc[df['wave'] == 7]

df_cz = df_9.loc[df_9['country_mod'] == 203]


cols = [
 'female',
 'age',
 'iv009_mod',
 'isced1997_r',
 'eduyears_mod',
 'mar_stat',
 'hhsize',
 'partnerinhh',
 'sphus',
 'chronic_mod',
 'casp',
 'bmi',
 'ep005_',
 'co007_',
 'thinc_m']


df_nan = df_cz[cols].mask(df_cz[cols] < 0, np.nan)

df_nan.isnull().sum().sort_values(ascending=False)


# %%
###############################################################################
# 3) Identifikace vzoru chybějících dat (MCAR, MAR, MNAR)
###############################################################################


mt = MCARTest(method="little")
mt.little_mcar_test(df_nan)

# %%

# Vypočítání matice phi_k
phi_k_mat = df_nan.phik_matrix(interval_cols=['age', 'thinc_m',
'bmi', 'eduyears_mod'])

print("Phi_k matice:")
print(phi_k_mat)

# Vizualizace
plt.figure(figsize=(9, 7))
sns.heatmap(phi_k_mat, annot=False, cmap="RdBu", center=0)
plt.title("Korelační matice phi_k v datovém souboru easySHARE")
plt.show()


# %%
# Vizualizace chybějících hodnot
msno.matrix(df_nan)


# %%
# Počet řádků bez NaN hodnot
num_rows_no_na = df_nan.dropna().shape[0]
print(f"Počet řádků bez NaN hodnot: {num_rows_no_na}")


# Řádky, kde chybí 'thinc_m' a zároveň jsou všechny hodnoty prediktorů vyplněné
missing_thinc_m_with_pred_filled = df_nan[df_nan['thinc_m'].isna() & df_nan[pred_cols].notna().all(axis=1)]

print(missing_thinc_m_with_pred_filled)
# %%

print((df_nan.isnull().sum()/len(df_nan)*100).sort_values(ascending=False).round(2))
#%%
msno.heatmap(df_nan)  # další možnost
# %%
msno.dendrogram(df_nan)
# %%

def little_mcar_test(df, alpha=0.05):
    """
    Performs Little's MCAR (Missing Completely At Random) test on a dataset with missing values.
    """
    data = df.copy()
    data.columns = ['x' + str(i) for i in range(data.shape[1])]
    data['missing'] = np.sum(data.isnull(), axis=1)
    n = data.shape[0]
    k = data.shape[1] - 1
    df = k * (k - 1) / 2
    chi2_crit = chi2.ppf(1 - alpha, df)
    chi2_val = ((n - 1 - (k - 1) / 2) ** 2) / (k - 1) / ((n - k) * np.mean(data['missing']))
    p_val = 1 - chi2.cdf(chi2_val, df)
    if chi2_val > chi2_crit:
        print(
            'Reject null hypothesis: Data is not MCAR (p-value={:.4f}, chi-square={:.4f})'.format(p_val, chi2_val)
        )
    else:
        print(
            'Do not reject null hypothesis: Data is MCAR (p-value={:.4f}, chi-square={:.4f})'.format(p_val, chi2_val)
        )


# Test MCAR
little_mcar_test(df_nan, 0.05)

# %%
# Zde jen ilustrativně ukážeme, že lze vypsat korelaci "missingness" s jinými
# sloupci, abychom posoudili, zda je nepřítomnost hodnot závislá na jiném sloupci.


# Na reálném datasetu by bylo potřeba podrobněji zkoumat:
# - existuje statisticky významná korelace mezi missingness a existující proměnnou?
# - je mechanismus chybění ovlivněn nepozorovanými faktory, kontextem atd.?

# Pro další práci předpokládejme, že se jedná o MAR (což je nejčastější reálný scénář),
# i když v praxi je toto nezbytné obhájit.

###############################################################################
# 4) Tradiční imputační techniky
###############################################################################

# V praxi je vhodné kombinovat několik metod a porovnat jejich výsledky.
# Napriklad pro numerické proměnné můžeme použít regresní imputaci, pro kategorické mód,
# a pro obě metody vícenásobnou imputaci (MICE). Dále můžeme vyzkoušet autoenkodér.

# Pro demonstraci použijeme jen několik sloupců z datasetu, abychom měli jednodušší
# implementaci autoenkodéru. Kategorická data by se musela zpracovat jinak (např. one-hot encodingem).

# Zde si vytvoříme kopii původního datasetu, abychom mohli postupně přidávat sloupce s imputacemi.


# Chteli bychom se podivat na strukturu dat, napriklad na rozdeleni hodnot jednotlivych promennych 
# a pripadne na korelace mezi nimi.


#########################################################################

# Zde bychom měli provést kontrolu, zda imputace proběhla správně.
# Např. u kategorických proměnných bychom měli zkontrolovat, zda se
# neobjevily nové hodnoty, které nebyly v původním datasetu.

# Pro numerické proměnné můžeme zkontrolovat, zda se nám nepoškodily distribuce,
# např. zda se neobjevily nové extrémní hodnoty, které nebyly v původním datasetu.

# %%
# 4d) Regresní imputace (příklad)

# 1) připrav prediktory (X) a cílovou proměnnou (y)

pred_cols = [
 'female',
 'age',
 'iv009_mod',
 'isced1997_r',
 'eduyears_mod',
 'mar_stat',
 'hhsize',
 'partnerinhh',
 'sphus',
 'chronic_mod',
 'casp',
 'bmi',
 'ep005_',
 'co007_'] # relevantní sloupce


# %%

###############################################################################
# 5) Vícenásobná imputace (IterativeImputer) 
#    - v Pythonu zjednodušeně MICE-like přístup
###############################################################################

# Vytvoříme menší podmnožinu dat, se kterými budeme MICE demonstrovat.
# Předpokládejme, že age i income jsou numerické a chceme je oba imputovat.
# od verze scikit-learn 0.20 je nutno aktivovat experimentální API

# Vybereme sloupce, které chceme v iterativní imputaci zapojit
cols_for_imputation = pred_cols

# Subset dataframe s vybranými proměnnými
df_subset = df_nan[cols_for_imputation].copy()

# Vytvoření imputeru
imputer = IterativeImputer(
    random_state=14,
    max_iter=10  # počet imputačních cyklů
)

# fit_transform = natrénuje (fit) a následně naplní (transform)
df_imputed_array = imputer.fit_transform(df_subset)

# Převedení zpět na DataFrame s původními názvy
df_imputed = pd.DataFrame(df_imputed_array, columns=cols_for_imputation)

# Místo, kde se uložily doplněné hodnoty
print(df_imputed.head(10))


# %%

# now we join the 'thinc_m' column to the imputed data
extracted_col = df_nan['thinc_m']

# Add the extracted column to the second DataFrame
df_imputed['thinc_m'] = df_nan['thinc_m'].values

# but the 
###############################################################################
# Ověření, jestli data jsou MAR

# Indikátor chybějícího příjmu
df_imputed['missing_thinc_m'] = df_imputed['thinc_m'].isna().astype(int)


X_stats = pd.get_dummies(df_imputed[pred_cols], drop_first=True)
X_stats = sm.add_constant(X_stats)  # přidá intercept
y_stats = df_imputed['missing_thinc_m']


logit_model = sm.Logit(y_stats, X_stats)
result = logit_model.fit(disp=False)
print(result.summary())

# %%
# Ověření, jestli data jsou MNAR pomocí citlivostní analýzy

# Uložíme si informaci, kde byl původně missing `thinc_m`
notmissing_idx = df_imputed['thinc_m'].notna()

# %%
# Například posun od -20 % do +20 % v krocích po 10 %
# shifts = [-0.2, -0.1, 0, 0.1, 0.2]

# Případně jemnější škálu:
shifts = np.linspace(-0.2, 0.2, 9)  # -20%, -15%, ..., 0, ..., 15%, 20%
import statsmodels.api as sm
# 3) Citlivostní analýza
#shifts = [-0.2, -0.1, 0, 0.1, 0.2]

results = []

for shift in shifts:
    # Kopie
    df_scenario = df_imputed.copy()
    
    # Posunout imputované `thinc_m` o shift% jen tam, kde původně chybělo
    df_scenario.loc[notmissing_idx, 'thinc_m'] *= (1 + shift)
    
    # Např. spočítat průměr
    mean_thinc_m = df_scenario['thinc_m'].mean()
    
    # Jednoduchá regrese `thinc_m ~ age + eduyears_mod`
    X = sm.add_constant(df_scenario[pred_cols])
    y = df_scenario['thinc_m']
    
    model = sm.OLS(y, X, missing='drop').fit()
    r2 = model.rsquared
    coef_age = model.params.get('age', np.nan)
    
    results.append({
        'shift_%': shift * 100,
        'mean_thinc_m': mean_thinc_m,
        'r2': r2,
        'coef_age': coef_age
    })

df_sensitivity = pd.DataFrame(results)
print(df_sensitivity)

# %%

df_train = df_imputed.dropna(subset=['thinc_m'])  

# Regresní imputace pro thinc_m

# a) Identifikuj trénovací a imputovací řádky, ale indexy ponech tak, jak jsou
  # tady indexy zůstanou např. [0,2,3,5]
df_missing = df_imputed[df_imputed['thinc_m'].isna()]       # [1,4], pokud tam pred_cols nemají NaN

# b) Ověřit, že pred_cols v df_missing jsou bez NaN
df_missing = df_missing.dropna(subset=pred_cols)  # furt indexy [1,4]

# c) Trénink regrese
lin_reg = LinearRegression()
lin_reg.fit(df_train[pred_cols], df_train['thinc_m'])
lin_reg.score(df_train[pred_cols], df_train['thinc_m'])

# d) Predikce
y_pred = lin_reg.predict(df_missing[pred_cols])

# e) Přiřazení do původního df: teď indexy sedí (df_missing.index = [1,4])
#df_imputed.loc[df_missing.index, 'thinc_m'] = y_pred

# %%

# 2) Definice X (prediktorů) a y (cílové proměnné)
X = df_train[pred_cols]
y = df_train['thinc_m']

# 3) Přidání konstanty (interceptu) - statsmodels to vyžaduje ručně
X = sm.add_constant(X)  # sloupec 'const' bude intercept

# 4) Vytvoření a odhad modelu (Ordinary Least Squares)
model = sm.OLS(y, X)      # definice modelu
results = model.fit()      # fitnutí
#print(results.summary())   # shrnutí s p-hodnotami, R2, atd.

# Můžeš také získat parametry samostatně
#print("\nParametry modelu:")
#print(results.params)  # koeficienty (intercept, age, female, eduyears_mod)

#print("\nR^2 =", results.rsquared)


# 5) Robustní odhad kovarianční matice
robust_cov = results.get_robustcov_results(cov_type='HC3')
print(robust_cov.summary())


# %%


##############################################################################
# 1) Připrava funkce pro umělé maskování a validaci
##############################################################################¨


def mask_thinc_m_for_validation(df, col='thinc_m', fraction=0.2, random_state=42):
    """
    U vybraného sloupce (col) uměle nahradí fraction podílu existujících hodnot 
    za NaN (pokud reálně nechybí). Sloupec s původními hodnotami se uloží do
    col+'_orig' pro validaci.
    """
    df_out = df.copy()
    np.random.seed(random_state)

    # Vybereme indexy, kde reálně nechybí col
    not_missing_mask = df_out[col].notna()
    idx_full = df_out[not_missing_mask].index
    idx_full_shuffled = np.random.permutation(idx_full)


    # Odhad parametrů logaritmického rozdělení ze stávajících hodnot 'thinc_m'
    thinc_m_values = df_out.loc[not_missing_mask & (df_out[col] > 0), col]
    shape, loc, scale = scipy.stats.lognorm.fit(thinc_m_values, floc=0)

    # Kolik z nich uměle "skryjeme" (uděláme NaN)?
    n_to_mask = int(len(idx_full_shuffled) * fraction)

    # Generování pravděpodobností pro každý index na základě logaritmicko-normálního rozdělení
    probabilities = scipy.stats.lognorm.pdf(df_out.loc[idx_full_shuffled, col], shape, loc, scale)
    probabilities /= probabilities.sum()  # Normalize to sum to 1
    
    # Kolik z nich uměle "skryjeme" (uděláme NaN)?
    mask_idx = np.random.choice(idx_full_shuffled, size=n_to_mask, replace=False, p=probabilities)

    # Uložíme si originální hodnoty a pak je skryjeme
    df_out[col + '_orig'] = df_out[col]
    df_out.loc[mask_idx, col] = np.nan

    return df_out

def compute_metrics_and_plot(df_after, col='thinc_m'):
    """
    Spočítá RMSE a MAE na uměle maskovaných záznamech:
      - y_true = col+'_orig'
      - y_pred = col
    Vykreslí histogram a scatterplot (s čarou y = x).
    """
    # Odfiltrujeme řádky, kde byl col původně znám
    # a nyní by mohl být imputován.
    df_eval = df_after.dropna(subset=[col + '_orig']).copy()

    # y_true = původní hodnota, y_pred = nová (imputovaná)
    y_true_all = df_eval[col + '_orig']
    y_pred_all = df_eval[col]

    # Chceme se zaměřit na řádky, které byly opravdu "uměle" schovány 
    # (tj. doopravdy imputovány).
    changed_mask = (df_eval[col] != df_eval[col + '_orig'])
    df_eval_changed = df_eval[changed_mask]
    if df_eval_changed.empty:
        print("Nebyla maskována žádná hodnota pro validaci (nebo imputace shodou okolností vrátila původní hodnoty).")
        return

    y_true = df_eval_changed[col + '_orig']
    y_pred = df_eval_changed[col]

    # Výpočet metrik
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")

    # 1) HISTOGRAM: porovnání rozdělení původních (y_true) a imputovaných (y_pred)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)  # rozdělení na dva subplots
    sns.histplot(y_true, color='blue', alpha=0.5, bins=30, label='Původní (uměle skryté) hodnoty', kde=True)
    sns.histplot(y_pred, color='red', alpha=0.5, bins=30, label='Imputované hodnoty', kde=True)
    plt.title(f"Srovnání rozdělení pro proměnnou {col}\n (RMSE={rmse:.2f}, MAE={mae:.2f})")
    plt.legend()
    plt.xlabel("Čisté roční příjmy domácnosti v EUR")
    plt.ylabel("Absolutní četnost")


    # 2) SCATTER PLOT: y_true vs y_pred, plus y=x
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.4, color='green', label='Body (původní vs imputované)')
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
    plt.title(f"Bodový graf: {col}\n skutečné vs. imputované hodnoty ")
    plt.xlabel("Původní (uměle skryté) hodnoty")
    plt.ylabel("Imputované hodnoty")
    plt.legend()

    plt.tight_layout()
    plt.show()

##############################################################################
# 2) Umělé maskování a příprava
##############################################################################

# Předpokládejme, že df_imputed má všechny prediktory vyplněné,
# ale 'thinc_m' je občas reálně NaN.
# pro validaci schováme 20 % těch, co reálně nechyběly.
df_val = mask_thinc_m_for_validation(df_imputed, col='thinc_m', fraction=0.2, random_state=42)

# df_val nyní obsahuje:
#   - 'thinc_m' (místy reálně missing + 20% uměle missing)
#   - 'thinc_m_orig' pro řádky, kde nebylo missing a nevybral ho random

##############################################################################
# 3) REGRESNÍ IMPUTACE
##############################################################################

# a) Rozdělíme df_val na:
#    - trénovací část (kde thinc_m NENÍ NaN) => fitneme regresi
#    - chybějící část => predikujeme
df_regress = df_val.copy()

train_mask = df_regress['thinc_m'].notna()  # trénink: známý příjem
test_mask  = df_regress['thinc_m'].isna()   # "test": chybějící příjem

# X sloupce = všechny prediktory, 
#   v df_imputed by měly být vyplněné, např. ['age', 'female', 'eduyears_mod', ...]
#   Dej tam, které opravdu chceš použít:  # příklad

X_train = df_regress.loc[train_mask, pred_cols]
y_train = df_regress.loc[train_mask, 'thinc_m']

X_test = df_regress.loc[test_mask, pred_cols]

# b) Natrénovat lineární regresi
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# c) Predikovat
y_pred = reg_model.predict(X_test)
df_regress.loc[test_mask, 'thinc_m'] = y_pred

# Calculate R^2 for the regression model
r2 = r2_score(df_train['thinc_m'], lin_reg.predict(df_train[pred_cols]))
print(f"R^2 for the regression model: {r2:.2f}")

# d) Vyhodnocení
print("=== Výsledky REGRESNÍ imputace ===")
compute_metrics_and_plot(df_regress, 'thinc_m')

##############################################################################
# 4) kNN IMPUTACE
##############################################################################
# V kNNImputeru bude "thinc_m" i prediktory v jedné matici, aby 
# se kNN mohlo řídit podobností (včetně 'age', 'female', 'eduyears_mod').
# Minimálně 'thinc_m' a vybrané prediktory tam musí být.

df_knn = df_val.copy()

cols_knn = cols  # ... 

def gower_distance(X, Y=None, **kwargs):
    if Y is None:
        Y = X
    individual_variable_distances = []
    for col in X.columns:
        if X[col].dtype == 'object':
            individual_variable_distances.append(pd.get_dummies(X[col]).values)
        else:
            scaler = MinMaxScaler()
            individual_variable_distances.append(scaler.fit_transform(X[[col]]))
    individual_variable_distances = np.hstack(individual_variable_distances)
    individual_variable_distances_Y = []
    for col in Y.columns:
        if Y[col].dtype == 'object':
            individual_variable_distances_Y.append(pd.get_dummies(Y[col]).values)
        else:
            scaler = MinMaxScaler()
            individual_variable_distances_Y.append(scaler.fit_transform(Y[[col]]))
    individual_variable_distances_Y = np.hstack(individual_variable_distances_Y)
    return np.sqrt(((individual_variable_distances[:, None, :] - individual_variable_distances_Y[None, :, :]) ** 2).sum(axis=2))

# Calculate Gower's distance matrix
gower_distances = gower_distance(df_knn[cols_knn])

# Fit the kNN model using Gower's distance
knn_imputer = KNNImputer(n_neighbors=23, metric='precomputed')
df_knn_imputed = knn_imputer.fit_transform(gower_distances)

# Convert the imputed array back to a DataFrame
df_knn[cols_knn] = df_knn_imputed


mat_before = df_knn[cols_knn].values  # numpy array

mat_after = knn_imputer.fit_transform(mat_before)

df_knn[cols_knn] = mat_after

# Po kNNImputeru by 'thinc_m' už nemělo mít NA tam, kde prediktory existují.
# Vyhodnotíme
print("=== Výsledky kNN imputace ===")
compute_metrics_and_plot(df_knn, 'thinc_m')

##############################################################################
# KONEC
##############################################################################

