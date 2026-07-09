# Foto vs. Camp — Clastometria

App Streamlit per comparar dues sèries de mides de clast (eix-b), mesurades
per foto i per camp: descriptius, percentils, tests estadístics, models de
regressió (additiu i multiplicatiu), banda ±2σ, bootstrap i gràfics.

## Com publicar-la perquè els alumnes hi accedeixin des del navegador

1. **Puja aquests 3 fitxers a un repositori de GitHub** (`app.py`,
   `requirements.txt`, aquest `README.md`). Pot ser un repo públic o
   privat.
2. Ves a **https://share.streamlit.io** i inicia sessió amb el teu compte
   de GitHub (Streamlit Community Cloud és gratuït).
3. Clica **"New app"**, selecciona el repositori i el fitxer `app.py`.
4. Clica **Deploy**. En un minut tindràs una URL pública del tipus
   `https://el-teu-app.streamlit.app` que pots compartir amb els alumnes.

Cada cop que actualitzis `app.py` a GitHub (canvis de codi, nous gràfics,
etc.), l'app es redesplega automàticament — no cal tocar res més.

## Com fer-la servir un alumne

1. Obre la URL de l'app al navegador (mòbil o ordinador).
2. Enganxa les seves dades de FOTO en la primera caixa i les de CAMP en la
   segona (números separats per comes, espais o un per línia).
3. Els resultats i gràfics es generen automàticament.

## Provar-la en local abans de publicar-la

```bash
pip install -r requirements.txt
streamlit run app.py
```
