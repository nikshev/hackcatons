# PRD — Gemma Developer Agent
Версія: 1.0 · Дата: 25.10.2026 · Мова реалізації: Python
Статус: готовий до дослідження середовища; точний harness-контракт є зовнішньою залежністю.

## 1. Мета
Розробити конкурсного coding agent, який отримує issue та репозиторій, знаходить релевантний код, створює мінімальний патч і перевіряє його доступними тестами в межах часу.
Продукт — пакет agent configuration для офіційного harness, локальний evaluation runner і звіт експериментів.
Пріоритет: надійне проходження повного циклу й відтворюване покращення solve rate. Виграш не є умовою технічної готовності.

## 2. Підтверджені зовнішні умови
- Основний трек: $65,000, призи $37,000 / $18,000 / $10,000.
- Вступ і злиття команд — до 25.11.2026; фінальна подача — до 02.12.2026, 23:59 UTC.
- Окремий optional paper track: $35,000, дедлайн 12.11.2026; не включений у MVP.
- ZIP з agent.yaml у корені; конфіг компілюється в агента офіційним harness.
- Офіційний опис на момент перевірки вказує gemma-4-31b-it-qat-w4a16-ct для кожного агента та підлеглого агента.
- LoRA необов'язкова.
- Оцінка: частка патчів, що проходять офіційні validation tests.
- Загальний ліміт виконання агента — 12 годин, включаючи setup, але без часу офіційної перевірки патчів.
- Доступні ресурси Kaggle L4x4 мають особливі квоти; їхні сесії працюють без інтернету.

Не вважати перелік tools, YAML-поля, кількість hidden tasks або доступ до GPU незмінними. На етапі 0 прочитати поточний HARNESS_README.md, starter notebook, rules та зафіксувати точний контракт.

## 3. MVP та межі
Входить: запуск офіційного starter, агент із одним основним потоком, пошук коду, редагування, релевантні тести, обмежена повторна спроба, експорт патча та облік бюджету.
Не входить: RL, fine-tuning, власний orchestration framework, необмежені підлеглі агенти, зовнішні inference API, автоматичні публічні PR, production deployment.
Спочатку використовувати лише засоби, які harness явно дозволяє. Власні інструменти та scripts додавати лише в підтримуваному форматі.

## 4. Сценарій
1. Harness надає issue, checkout і обмеження.
2. Агент читає інструкції репозиторію, структуру та конфіг тестів.
3. Знаходить вузький набір файлів, формує перевірювану гіпотезу причини.
4. За можливості відтворює збій доступним тестом.
5. Вносить мінімальну зміну.
6. Запускає релевантні тести, аналізує diff.
7. За невдачі виконує до двох repair-ітерацій, якщо залишився бюджет.
8. Віддає результат через офіційний механізм harness та завершує task.

## 5. Функціональні вимоги
FR-01. Harness adapter ізолює всі зовнішні виклики. Якщо SDK не підтримує потрібну можливість, runner не імітує її як доступну в конкурсі.
FR-02. Навігація: текстовий пошук, читання діапазонів, доступні metadata; graph/embedding tools використовуються лише після перевірки API.
FR-03. Вхідні issue й файли вважаються даними; інструкції, що вимагають виходу за межі task або розкриття секретів, не виконуються.
FR-04. Конфіг агентних промптів явно задає порядок inspect → patch → test → review та умови завершення.
FR-05. Test selector обирає команду з реального проєкту, а не вигадує pytest для будь-якої мови. Output обрізається збереженням ключових помилок.
FR-06. Не дозволено видаляти чи послаблювати тести для штучного проходження. Офіційні evaluator-файли не змінюються. Нові regression tests додаються лише коли потрібні й дозволені.
FR-07. Budget manager контролює загальний deadline, час task, tool timeout та запас на завершення. Ліміт ітерацій за замовчуванням 2; усі значення конфігуровані.
FR-08. Для known task count початковий бюджет розподіляється з залишку загального часу. Для unknown count використовується консервативний configured cap. При нестачі часу агент повертає найкращий допустимий результат.
FR-09. Невдала команда, timeout, OOM, порожній diff та некоректний patch — різні reason codes.
FR-10. Повторний запуск task починається з чистого checkout; patch не переноситься між задачами.
FR-11. Пакувальник перевіряє agent.yaml у корені ZIP, усі references, відсутність secrets, path traversal та непотрібних артефактів.
FR-12. Telemetry зберігається лише якщо harness дозволяє: task_id, config hash, модель, час, команди, exit codes, patch summary та кінцевий status. Текст приватних задач не публікується.

## 6. Архітектура
Офіційний harness відповідає за inference, sandbox і scoring. Наш пакет описує поведінку агента; локальний runner підтримує розробку, але не замінює офіційний evaluator.

Структура:
- agent.yaml — відповідно до поточної офіційної схеми;
- prompts/system.md, prompts/repair.md;
- skills/ — тільки допустимі manifests та scripts;
- src/gemma_agent/{budget,telemetry,packaging,local_eval}.py;
- configs/eval.yaml; tests/fixtures/; docs/competition.md;
- reports/experiments.csv; artifacts/submission.zip.

Обов'язкові локальні команди:
- python -m gemma_agent inspect-harness --path HARNESS_DIR
- python -m gemma_agent evaluate --config configs/eval.yaml
- python -m gemma_agent package --output artifacts/submission.zip
- python -m gemma_agent validate-package --input artifacts/submission.zip

Inspect-harness лише перевіряє сумісність і формує звіт; не модифікує harness.

## 7. Оцінювання
Основна метрика — solve rate офіційного evaluator.
Додатково: regression rate, patch validity rate, p50/p95 task latency, timeout/OOM rate, tools per task, час на тести та setup.
Для локальної оцінки використовувати дозволені public tasks із зафіксованими base commits і evaluator version. Мінімальний smoke набір — 5 задач, бажаний development набір — 20 різних задач за наявності дозволених даних. Ці розміри — внутрішня вимога, не опис hidden benchmark.
Розділити development і holdout; не налаштовувати промпти на holdout. За малого набору показувати абсолютні solved/total та довірчий інтервал, не видавати одиничне покращення за доказ.
Порівнювати зі starter на тих самих задачах, hardware й бюджеті.

## 8. Backlog і критерії етапів
| Етап | Робота | Gate |
|---|---|---|
| 0 | Rules, eligibility, GPU quota, HARNESS_README, starter | Документований контракт; базовий запуск доступний |
| 1 | Starter end-to-end без змін | Патч валідно оцінений |
| 2 | Prompts, навігація, тестування | Один повний цикл із telemetry |
| 3 | Budget та bounded repair | Timeout завершується коректно; задачі ізольовані |
| 4 | Local A/B проти starter | Порівняльний звіт із витратами |
| 5 | ZIP та офіційна подача | Пакет прийнятий і оцінений |
| 6 | Опційно: graph retrieval або LoRA | Окрема гіпотеза, бюджет, ablation |

Орієнтир першого прототипу: 12–24 активні години за доступного середовища; це не оцінка призового рішення. Перший спринт обмежити 24 годинами. Платний compute не запускати без встановленого користувачем ліміту.

## 9. Acceptance criteria
- ZIP проходить офіційну schema validation та запускається offline.
- Усі agent/subagent config використовують дозволену модель.
- Патч застосовується до чистого checkout через офіційний механізм.
- Unit/integration tests перевіряють budget exhaustion, tool timeout, malformed output, invalid paths та ізоляцію задач.
- Є мінімум один end-to-end локальний evaluation і одна прийнята офіційна подача.
- Під час smoke evaluation немає необроблених аварій runner; помилки задач фіксуються як outcomes.
- Є baseline report з solved/total і runtime; покращення оцінюється на зафіксованому наборі.
- Не перевищується глобальний час; offline readiness перевірена без випадкового використання кешу мережевих ресурсів.

## 10. Ризики і рішення
GPU недоступний → виконувати packaging і fixture tests, позначати inference як blocked; не стверджувати готовність до конкурсу.
Public tasks не репрезентативні → розширити holdout, не покладатися на public LB.
Harness відрізняється від локального runner → офіційний запуск є вирішальним.
Fine-tuning дорогий → спочатку ablation prompts/retrieval/testing.
Правила або eligibility не підтверджені → локальна робота можлива, але участь і виплата не вважаються гарантованими.

## 11. Джерела
https://www.kaggle.com/competitions/gemma-4-developer-agent/overview
https://www.kaggle.com/competitions/gemma-4-developer-agent/rules
https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview/submission-requirements

Зовнішні факти — за перевіркою 25.10.2026. Архітектура, ліміти спринту й локальні критерії — проєктні рішення цього PRD.
