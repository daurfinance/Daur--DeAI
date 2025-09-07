# Blockchain Quickstart (Daur-AI)

## 1. Установка зависимостей

```bash
npm install -g truffle
npm install -g ganache
```

## 2. Запуск локального блокчейна

```bash
ganache --port 8545
```

## 3. Компиляция и деплой контракта

```bash
cd blockchain
truffle compile
truffle migrate --network development
```

## 4. Запуск тестов смарт-контракта

```bash
truffle test ../tests/TaskRegistry.test.js
```

## 5. Интеграция с Python-нодами
- Для взаимодействия используйте web3.py или аналогичный SDK.
- Пример вызова: создание задачи, отправка partial, агрегация.

---

**Примечание:**
- Контракт: `TaskRegistry.sol`
- Тесты: `tests/TaskRegistry.test.js`
- Конфиг: `blockchain/truffle-config.js`

Для вопросов и багов — пишите в Issues репозитория.
