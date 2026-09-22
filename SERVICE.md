# HydroWatch Amur — сервис мониторинга затопления

REST API + лёгкая Leaflet-карта поверх масок предсказаний.

## Запуск

```bash
python service.py
# или
uvicorn service:app --host 0.0.0.0 --port 8000
```

Открыть `http://127.0.0.1:8000/` — карта с переключаемыми слоями `flood / water_pre / water_peak`.

## Эндпоинты

| Метод | URL | Что возвращает |
|---|---|---|
| GET  | `/health` | статус и пути к данным |
| GET  | `/pairs` | список 11 пар (id, AOI, событие, даты, есть ли оптика) |
| GET  | `/areas/{pair_id}` | предсказанные и эталонные площади water_pre/water_peak/flood в га |
| GET  | `/contours/{pair_id}?layer=flood` | GeoJSON контуры воды (CRS EPSG:32652) |
| POST | `/report` | сводный отчёт: gain, flood_share_of_aoi, by_layer_in_roi |
| GET  | `/layer/{pair_id}/{layer}.png` | бинарная маска как PNG для оверлея на карте |
| GET  | `/` | HTML-карта с Leaflet |

### `/report` тело запроса

```json
{
  "pair_id": "flood_2019_07_amur__belogorsk",
  "bbox": [128.0, 51.0, 128.5, 51.3],
  "min_pixels": 5
}
```

`bbox` — в WGS84. Без bbox эндпоинт всё равно возвращает площади и прирост, без разбивки по ROI.

## Где что лежит

- `out/predictions/<pair_id>_flood.tif` — маски затопления (uint8, 0/1)
- `out/submission.csv` — предсказанные площади в га (читается сервисом через `/areas`)
- Маски `water_pre.tif`, `water_peak.tif` кладутся `infer_combine.py` рядом с `flood.tif`

## Площади считаются тем же кодом, что и сабмит

`service.py` использует `hw.hectares()` и `hw.geo()` из общего модуля — рассинхрона с `infer_combine.py` нет.

## Пример curl

```bash
# Список пар
curl http://127.0.0.1:8000/pairs

# Площади по паре
curl http://127.0.0.1:8000/areas/flood_2019_07_amur__belogorsk

# GeoJSON контуры
curl 'http://127.0.0.1:8000/contours/flood_2019_07_amur__belogorsk?layer=flood&min_pixels=20'

# Отчёт по bbox
curl -X POST http://127.0.0.1:8000/report \
  -H 'Content-Type: application/json' \
  -d '{"pair_id": "flood_2019_07_amur__belogorsk", "bbox": [128.0, 51.0, 128.5, 51.3]}'

# Маска как PNG (для оверлея на карте)
curl -o flood.png http://127.0.0.1:8000/layer/flood_2019_07_amur__belogorsk/flood.png
```
