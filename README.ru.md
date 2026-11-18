# lb2_kurenko — классификация поверхностей по данным датчиков

**Бинарная классификация типа поверхности по измерениям робота с помощью PyTorch, Optuna и сравнительных экспериментов.**

[English](README.md) | Русский

Проект исследует возможность различать класс поверхности по измерениям датчиков робота с помощью нейронной сети. Целевая метка формируется из столбца `Type`: **класс 1 — `Type == 2`**, **класс 0 — остальные типы**. В репозитории есть обучение и тестирование, сохранённая модель, нормализатор, Excel-таблицы и графики влияния разных параметров.

![Сравнение функций активации](activation_analysis.png)

## Возможности

- **Бинарный классификатор на PyTorch:** MLP со слоями `nn.Linear`, функциями активации и dropout.
- **15 признаков датчиков:** токи `I1`–`I3`, гироскоп `gx`–`gz`, акселерометр `ax`–`az`, напряжения `V1real`–`V3real`, параметры `N1`–`N3`.
- **Подбор Optuna:** ширина и глубина сети, скорость обучения, batch size, dropout и активация.
- **Работа с дисбалансом классов:** взвешенная cross-entropy и отдельные исследования балансировки.
- **Оценка классификатора:** F1, accuracy, матрица ошибок, classification report, ROC/AUC в `testing.py`.
- **Сравнительные эксперименты:** функции активации, число эпох, оптимизаторы, слои, нейроны, random state, нормализация и балансировка.

## Требования

Рекомендуется Python 3.10+; нужны `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `joblib`, `openpyxl`, `imbalanced-learn` (последняя библиотека импортируется в экспериментальных скриптах).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy pandas scikit-learn matplotlib optuna joblib openpyxl imbalanced-learn
```

## Данные

| Файл | Назначение |
|---|---|
| `raw/train.xlsx`, `raw/test.xlsx` | Исходные табличные обучающие и тестовые данные |
| `train_augmented.xlsx` | Подготовленная выборка для `main.py` |
| `train_imbalanced.xlsx`, `train_unnormalized.xlsx`, `train_unsorted.xlsx` | Варианты для отдельных экспериментов |
| `scaler_v1_features.pkl` | Сохранённый нормализатор для `testing.py` |
| `model.pt` | Сохранённые параметры нейросети |

`main.py` загружает `train_augmented.xlsx`. Подготовка данных дополнительно отражена в `preparation_dataset.ipynb`. Перед строгим сравнением результатов важно проверить, как формировались отдельные варианты датасета.

## Обучение и проверка

Из корня проекта:

```bash
python main.py        # Optuna, обучение итоговой модели, model.pt
python testing.py     # загрузка model.pt и scaler_v1_features.pkl, тест raw/test.xlsx
```

**Замечание:** `main.py` напрямую использует подготовленный датасет, тогда как `testing.py` нормализует сырой тестовый набор сохранённым scaler. Перед интерпретацией метрик и переносом модели проверьте согласованность преобразований на обучении и инференсе.

## Эксперименты и графики

| Скрипт | Исследование | Результат |
|---|---|---|
| `activation_analysis.py` | Функции активации | [`activation_analysis.png`](activation_analysis.png) |
| `epochs_analysis.py` | Число эпох | [`epochs_analysis.png`](epochs_analysis.png) |
| `layers_analysis.py` | Глубина сети | [`layers_analysis.png`](layers_analysis.png) |
| `neurons_analysis.py` | Число нейронов | [`neurons_analysis.png`](neurons_analysis.png) |
| `optimizer_analysis.py` | Оптимизаторы | [`optimizer_analysis.png`](optimizer_analysis.png) |
| `random_state_analysis.py` | Инициализация / разбиение | [`random_state_analysis.png`](random_state_analysis.png) |

Скрипты запускаются отдельно после установки зависимостей, например `python activation_analysis.py` из корня проекта.

## Структура репозитория

```text
main.py                         обучение и подбор Optuna
testing.py                      проверка сохранённой модели
preparation_dataset.ipynb       подготовка данных
train.ipynb, test.ipynb         исследовательские ноутбуки
*_analysis.py                   изучение параметров сети и обучения
*_experiment.py                 эксперименты с данными
raw/                            исходные Excel-файлы
train_*.xlsx                    подготовленные датасеты
model.pt, scaler_v1_features.pkl сохранённые артефакты
*.png                           графики экспериментов
```

## Статус и ограничения

- Метка получена из численного `Type`; детальная физическая интерпретация всех типов поверхности в архиве не документирована.
- Во время Optuna и итоговой оценки `main.py` использует одно разбиение `train_augmented.xlsx`. Качество на независимом тестовом наборе лучше сообщать отдельно.
- Есть сохранённая модель, но нет универсального пакета инференса или готового ROS-узла.
- В переданном архиве нет файла с лицензией проекта.
