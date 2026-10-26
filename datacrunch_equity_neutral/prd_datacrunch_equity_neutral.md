# PRD — DataCrunch Equity Market Neutral #2
Версія: 1.0 · Дата: 25.10.2026 · Мова реалізації: Python
Статус: готовий до старту baseline; submission adapter уточнюється за поточним quickstarter.

## 1. Мета
Створити відтворюваний pipeline прогнозування відносної майбутньої дохідності акцій для DataCrunch #2: отримання й аудит даних → часова валідація → навчання → локальна перевірка контракту → cloud submission → спостереження за live-результатами.
Користувач — один data engineer; pipeline має підтримувати короткі робочі сесії й поновлення після переривання.
Ціль MVP — технічно валідна модель і перша прийнята cloud-подача. Прогнозний сигнал не означає готової торгової стратегії.

## 2. Зовнішній контракт
За офіційною документацією:
- Щотижнева cross-sectional задача для ліквідних американських акцій.
- moon позначає послідовний тижневий timestamp.
- id не є стабільним ідентифікатором акції між moon; заборонено будувати історію конкретної акції лише за ним.
- Поточна описана метрика — Pearson correlation прогнозу й target на останній даті.
- Пул — 1,000 USDC на тиждень, не одноразові $52,000.
- Нижче медіани рейтингу виплат немає; вище медіани вага винагороди пропорційна 20-му степеню нормованого перевищення.
- Поточна документація вказує 15 годин CPU або GPU compute на тиждень, reset у неділю о 12:00 UTC.
- Run має завершитися до неділі 12:00 UTC для врахування цього тижня.
- Target визріває приблизно через п'ять тижнів; нові live-оцінки з'являються із затримкою.
- Quickstarter вимагає train та infer; точні сигнатури й output contract потрібно прочитати в поточному SDK.

Етап 0: перевірити доступність участі й виплат, актуальні rewards, quota, дані, target, поточний SDK і license. Не переносити Spearman зі старого DataCrunch Competition у #2 як основну метрику.

## 3. Обсяг
MVP: один актуальний dataset, baseline model, leakage-aware walk-forward evaluation, platform adapter, контроль ресурсів, артефакти експериментів та ручний запуск submission.
Після MVP: простий ensemble, локально заплановані щотижневі запуски за окремим рішенням користувача, dashboard за потреби.
Не входить: торгівля реальними коштами, брокерські ключі, staking, складний backtester зі штучно відновленими цінами, Kafka/Dagster/MLflow server, платна інфраструктура.

## 4. Вимоги до даних
FR-01. Одержувати дані лише дозволеним способом із поточної платформи. Зберігати immutable raw та manifest із checksum, schema version, source і fetched_at.
FR-02. Перевіряти типи moon/id/target/features, унікальність ключа відповідно до реального контракту, missing/inf, доступні дати й розподіли.
FR-03. Використовувати feature allowlist; виключати target, metadata та майбутню інформацію.
FR-04. Не припускати, що unresolved target дорівнює -1 або null: правило визначається актуальною документацією та перевіряється окремо. Такі targets виключати з train/evaluation.
FR-05. Не join-ити акції між тижнями за id, якщо немає підтвердженого стабільного ключа.
FR-06. Data report виявляє зміни schema, missingness, feature distribution і кількості рядків на moon. Пороги налаштовуються після першого audit.

## 5. Часова валідація
Використовувати expanding або rolling walk-forward; random row split заборонений як основна валідація.
Кожен fold має train_end, validation_start/end, target horizon та availability lag.
Вилучити train observations, чиї target intervals перекривають validation. Офіційний опис переходу до dataset #2 вказує embargo чотири moon; реалізація повинна врахувати фактичну доступність target, включаючи додатковий lag.
За умови перевірки weekly contract консервативний початковий gap — п'ять moon; це внутрішнє рішення, а не заміна офіційних правил. Зміна gap обґрунтовується в docs/validation.md.
Усі preprocessing transforms fit лише на train. Cross-sectional transforms на поточному moon дозволені тільки якщо весь цей cross-section доступний на inference.
Останній зрілий часовий відрізок — holdout; на ньому не налаштовувати параметри.

## 6. Моделі й метрики
Перший набір:
1. Constant predictor як технічний negative control; Pearson для нульової дисперсії undefined, не перетворювати мовчки на 0.
2. Ridge regression зі scaler та missing-value policy.
3. Одна gradient boosting модель: LightGBM за підтримки середовища, інакше сумісна альтернатива.
Гіперпараметри й seeds у YAML. GPU не обов'язкова залежність baseline.

Основна локальна метрика — Pearson по кожному validation moon, потім mean/median та розподіл; точний platform aggregation відтворити з документації.
Додаткові діагностики: Spearman, standard deviation IC, частка moon із позитивним IC, найгірший fold, train/inference runtime, peak RAM.
Не називати ці метрики Sharpe або portfolio return. Для реального PnL потрібні інші дані, costs і торгові припущення.
Фіксувати кількість експериментів; для оцінки невизначеності можна застосувати block bootstrap часових moon, а не незалежний bootstrap рядків.

## 7. Архітектура й команди
Стек: Python, pandas/Polars, pyarrow, numpy, scikit-learn, optional LightGBM, pytest; lockfile після перевірки сумісності SDK.
Локальні metadata — JSON/Parquet, без окремого сервера.
Структура:
- src/datacrunch/{data,audit,splits,features,train,evaluate,predict,platform_adapter,monitor}.py
- configs/baseline.yaml; tests/fixtures/; docs/competition.md; docs/validation.md
- artifacts/RUN_ID/; reports/; scripts/

CLI:
- python -m datacrunch audit --config configs/baseline.yaml
- python -m datacrunch backtest --config configs/baseline.yaml
- python -m datacrunch train --config configs/baseline.yaml
- python -m datacrunch validate-adapter --artifact ARTIFACT_DIR
- python -m datacrunch prepare-submission --artifact ARTIFACT_DIR

Backtest тут означає історичну перевірку прогнозів, не симуляцію фінансової дохідності.
Platform adapter реалізує train/infer у точній актуальній сигнатурі SDK; CLI її не підміняє.
Run manifest: dataset hash/version, git commit, config hash, split boundaries, feature list, preprocessing, модель, training cutoff, metrics, resource usage.
Фінальний artifact включає і preprocessing, і модель. На inference немає fit, звернення до target або навчання на майбутніх даних.

## 8. Експлуатація
- Ключі лише через environment variables; не логувати токени.
- Атомарний запис артефактів; завершений run відрізняється від partial.
- Ідемпотентний запуск із тим самим run/config не дублює cloud submission.
- Помилка даних зупиняє submission; немає мовчазного fallback на випадкові прогнози.
- Внутрішній ресурсний поріг задається конфігом і не перевищує актуальну квоту; залишити запас на повторну перевірку.
- До встановлення scheduler достатньо checklist ручного щотижневого запуску.
- Model registry містить candidate/submitted/accepted/scored states та platform run_id.
- Live results прив'язуються до prediction moon і expected maturity date. Незрілий результат не позначається невдачею.
- Credentials або доступ до платформи відсутні → local-ready, external-blocked; не імітувати успішну подачу.

## 9. Етапи
| Етап | Результат | Gate |
|---|---|---|
| 0 | SDK/rules/data contract | Реальні sample й quickstarter доступні |
| 1 | Audit та manifests | Немає нерозв'язаних schema помилок |
| 2 | Walk-forward та Ridge | Fold-метрики й leakage tests |
| 3 | Boosting, порівняння | Фіксовані splits, явні витрати |
| 4 | Adapter та clean-room inference | Локальний SDK test проходить |
| 5 | Перша cloud-подача | Платформа прийняла run |
| 6 | Live observation | Зрілі результати прив'язані до run |

Орієнтир MVP: 8–16 активних годин за доступних даних та SDK; live scoring потребує окремого календарного очікування. Після 16 годин — ревізія блокерів до розширення scope.

## 10. Acceptance criteria
- Відтворюються data audit, walk-forward, training та offline inference.
- Tests доводять відсутність часових overlap і fit на validation; boundary cases включають unresolved targets.
- Constant prediction, NaN/inf, неправильна довжина й переплутаний порядок рядків обробляються явно.
- Adapter повертає predictions у точному форматі й порядку SDK.
- Після перезапуску не втрачається зв'язок model → dataset → split → submission.
- Є reports для Ridge та boosting; числова quality target встановлюється після першого baseline.
- Є одна прийнята cloud submission; до визрівання targets статус live quality — pending.
- Позитивна validation correlation не видається за гарантований прибуток чи виплату.

## 11. Go/no-go
Продовжувати дослідження, якщо модель стабільна за folds, випереджає Ridge на фіксованому holdout із прийнятними витратами й має підтверджену eligibility. Якщо gains залежать від одного moon, зупинити нарощування складності й перевірити split/data.
Перші live-результати порівнювати з історичними очікуваннями, але не робити висновків про стабільний дохід за одним тижнем.

## 12. Джерела
https://hub.crunchdao.com/competitions/datacrunch-2
https://docs.crunchdao.com/competitions/competitions/datacrunch-2
https://github.com/crunchdao/docs/blob/master/competitions/competitions/datacrunch-2/from-dataset-1-to-dataset-2.md

Зовнішні умови — за перевіркою 25.10.2026. Архітектура, gap за замовчуванням, бюджет часу й acceptance criteria — запропоновані проєктні вимоги.
