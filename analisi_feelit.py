
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory


ETICHETTE = {
    "sentiment": ("negative", "positive"),
    "emozione": ("anger", "fear", "joy", "sadness"),
}
CLASSI = ETICHETTE["sentiment"] + ETICHETTE["emozione"]
COLONNE_SCORE = [f"score_{classe}" for classe in CLASSI]
NOMI = {
    "negative": "Negativo", "positive": "Positivo",
    "anger": "Rabbia", "fear": "Paura", "joy": "Gioia", "sadness": "Tristezza",
}
COLORI = {
    "negative": "#AA4937", "positive": "#08796F",
    "anger": "#B34F3E", "fear": "#826498", "joy": "#E4AE48", "sadness": "#477DA7",
}
MODI = {"originale": "Testo originale", "storica": "Testo normalizzato"}
LESSICO_PREDEFINITO = Path(__file__).with_name("lessico_1700.csv")
LIMITE_MODELLO = 500  # Stesso limite, token speciali inclusi, del pacchetto feel-it.
ANNO_CSV = re.compile(r"([1-9][0-9]{3})(?:[.,]0+)?")
FINE_FRASE = re.compile(r"[.!?][»”\"')\]]*\s+|\n[^\S\r\n]*\n")


@dataclass(frozen=True)
class Riga:
    numero: int
    anno: int
    testo: str


def leggi_csv(percorso, codifica, separatore):
    # Il limite predefinito di csv può essere troppo basso per un fascicolo OCR.
    csv.field_size_limit(100_000_000)
    richieste = {"years", "transcription"}
    righe, anni_invalidi = [], []
    with percorso.open(encoding=codifica, newline="") as file:
        candidati = (",", ";", "\t") if separatore == "auto" else (separatore,)
        for sep in candidati:
            file.seek(0)
            lettore = csv.reader(file, delimiter=sep, strict=True)
            try:
                colonne = next(lettore, [])
            except csv.Error:
                continue
            intestazioni = [colonna.strip().casefold() for colonna in colonne]
            if richieste <= set(intestazioni):
                break
        else:
            raise ValueError("Servono le colonne 'years' e 'transcription' nell'intestazione del CSV.")
        if any(intestazioni.count(nome) != 1 for nome in richieste):
            raise ValueError("Le colonne 'years' e 'transcription' devono comparire una sola volta.")

        indice_anno = intestazioni.index("years")
        indice_testo = intestazioni.index("transcription")
        for numero, valori in enumerate(lettore, start=1):
            if not valori:
                continue
            if len(valori) != len(intestazioni):
                raise ValueError(f"Record CSV {numero}. Attesi {len(intestazioni)} campi, trovati {len(valori)}.")
            anno = ANNO_CSV.fullmatch(valori[indice_anno].strip())
            if anno is None:
                anni_invalidi.append(numero)
                continue
            righe.append(Riga(numero, int(anno[1]), valori[indice_testo]))
    return righe, anni_invalidi, sep


def chiave_lessico(testo):
    testo = unicodedata.normalize("NFC", testo).replace("’", "'")
    return " ".join(testo.casefold().split())


def carica_lessico(dati):
    lettore = csv.DictReader(io.StringIO(dati.decode("utf-8-sig")), delimiter=";")
    if not {"storico", "moderno"} <= set(lettore.fieldnames or ()):
        raise ValueError("Il lessico deve contenere le colonne 'storico' e 'moderno', separate da ';'.")
    regole = {}
    for riga in lettore:
        storico = chiave_lessico(riga["storico"] or "")
        moderno = unicodedata.normalize("NFC", (riga["moderno"] or "").strip())
        if not storico or not moderno or None in riga:
            raise ValueError(f"Regola lessicale incompleta o malformata alla riga {lettore.line_num}.")
        if storico in regole:
            raise ValueError(f"Forma storica duplicata nel lessico: {storico!r}.")
        regole[storico] = moderno

    alternative = []
    for termine in sorted(regole, key=len, reverse=True):
        parole = [re.escape(parola).replace("'", "['’]") for parola in termine.split()]
        alternative.append(r"[^\S\r\n]+".join(parole))
    schema = re.compile(r"(?<!\w)(?:" + "|".join(alternative) + r")(?!\w)", re.IGNORECASE) if alternative else None
    return regole, schema


def normalizza(testo, regole, schema):
    conteggi = Counter()
    if schema is None:
        return testo, conteggi

    def sostituisci(match):
        originale = match[0]
        forma = chiave_lessico(originale)
        moderno = regole.get(forma, originale)
        if originale.isupper():
            moderno = moderno.upper()
        elif originale[0].isupper():
            moderno = moderno[0].upper() + moderno[1:]
        if moderno != originale:
            conteggi[forma] += 1
        return moderno

    return schema.sub(sostituisci, unicodedata.normalize("NFC", testo)), conteggi


def segmenta(testo, tokenizer, limite):
    codifica = tokenizer(testo, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    inizi = [inizio for inizio, _ in codifica["offset_mapping"]]
    frasi = [match.end() for match in FINE_FRASE.finditer(testo)]
    spazi = [match.end() for match in re.finditer(r"\s+", testo)]
    token, carattere = 0, 0
    while len(inizi) - token > limite:
        ultimo = inizi[token + limite]
        minimo = max(carattere + 1, inizi[token + limite // 2])
        taglio = max(ultimo, carattere + 1)
        for confini in (frasi, spazi):
            posizione = bisect_right(confini, ultimo) - 1
            if posizione >= 0 and confini[posizione] >= minimo:
                taglio = confini[posizione]
                break
        prossimo = bisect_left(inizi, taglio, lo=token + 1)
        yield testo[carattere:taglio], prossimo - token
        token, carattere = prossimo, taglio
    if carattere < len(testo):
        yield testo[carattere:], max(1, len(inizi) - token)


def carica_modelli(scelta):
    import torch
    from feel_it import EmotionClassifier, SentimentClassifier

    dispositivo = scelta
    if scelta == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if dispositivo == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA non è disponibile. Usa --dispositivo cpu.")
    if dispositivo == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    print(f"Caricamento dei due modelli FEEL-IT su {dispositivo}…", flush=True)
    modelli = {"sentiment": SentimentClassifier(), "emozione": EmotionClassifier()}
    for tipo, classificatore in modelli.items():
        classificatore.model.to(dispositivo)
        if not classificatore.tokenizer.is_fast:
            raise RuntimeError(f"Serve il tokenizer veloce FEEL-IT per il modello {tipo}.")
        if set(classificatore.model.config.id2label.values()) != set(ETICHETTE[tipo]):
            raise RuntimeError(f"Etichette FEEL-IT inattese per il modello {tipo}.")
    return modelli, dispositivo


def punteggi_modello(classificatore, testi, batch_size):
    import torch

    tokenizer, modello = classificatore.tokenizer, classificatore.model
    nomi = [modello.config.id2label[i] for i in range(modello.config.num_labels)]
    # Le finestre in eccesso vengono restituite e aggregate, anche con un lessico personalizzato.
    codifica = tokenizer(
        testi, truncation=True, max_length=LIMITE_MODELLO, padding=True,
        return_overflowing_tokens=True, return_tensors="pt",
    )
    proprietari = codifica.pop("overflow_to_sample_mapping").tolist()
    pesi = (codifica["attention_mask"].sum(1) - tokenizer.num_special_tokens_to_add()).clamp(min=1).tolist()
    totali = [Counter() for _ in testi]
    denominatori = [0] * len(testi)
    with torch.inference_mode():
        for inizio in range(0, len(proprietari), batch_size):
            fine = inizio + batch_size
            input_modello = {nome: valori[inizio:fine].to(modello.device) for nome, valori in codifica.items()}
            probabilita = modello(**input_modello).logits.softmax(dim=-1).cpu().tolist()
            for proprietario, peso, valori in zip(proprietari[inizio:fine], pesi[inizio:fine], probabilita):
                denominatori[proprietario] += peso
                for nome, valore in zip(nomi, valori):
                    totali[proprietario][nome] += peso * valore
    return [{nome: totale[nome] / peso for nome in nomi} for totale, peso in zip(totali, denominatori)]


def campi_punteggi(punteggi):
    return {
        **{tipo: max(classi, key=punteggi.__getitem__) for tipo, classi in ETICHETTE.items()},
        **{f"score_{classe}": round(punteggi[classe], 8) for classe in CLASSI},
    }


@dataclass
class MediaRiga:
    segmenti: int = 0
    token: int = 0
    totali: Counter = field(default_factory=Counter)
    sostituzioni: Counter = field(default_factory=Counter)

    def aggiungi(self, pesi, punteggi, cambi):
        self.segmenti += 1
        self.token += pesi
        self.sostituzioni.update(cambi)
        for classe, valore in punteggi.items():
            self.totali[classe] += valore * pesi


def tutti_i_segmenti(righe, tokenizer, limite, regole, schema):
    for riga in righe:
        if not riga.testo.strip():
            continue
        for indice, (originale, peso) in enumerate(segmenta(riga.testo, tokenizer, limite), 1):
            normalizzato, cambi = normalizza(originale, regole, schema)
            yield riga, indice, peso, {"originale": originale, "storica": normalizzato}, cambi


def scrittore(file, colonne):
    writer = csv.DictWriter(file, fieldnames=colonne, delimiter=";")
    writer.writeheader()
    return writer


def analizza(righe, modi, modelli, regole, schema, args, cartella):
    tokenizer = modelli["sentiment"].tokenizer
    segmenti = tutti_i_segmenti(righe, tokenizer, args.token_segmento, regole, schema)
    medie = defaultdict(MediaRiga)
    totale_segmenti = 0
    campi = ["riga_csv", "years", "modalita", "segmento", "token_originali", *ETICHETTE,
             *COLONNE_SCORE, "sostituzioni", "testo_originale", "testo_analizzato"]
    with (cartella / "risultati_segmenti.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = scrittore(file, campi)
        while blocco := list(islice(segmenti, args.batch_size)):
            testi = list(dict.fromkeys(varianti[modo] for _, _, _, varianti, _ in blocco for modo in modi))
            punteggi = {testo: {} for testo in testi}
            for classificatore in modelli.values():
                for testo, valori in zip(testi, punteggi_modello(classificatore, testi, args.batch_size)):
                    punteggi[testo].update(valori)

            for riga, indice, peso, varianti, cambi in blocco:
                for modo in modi:
                    testo = varianti[modo]
                    modifiche = cambi if modo == "storica" else {}
                    valori = punteggi[testo]
                    medie[riga.numero, modo].aggiungi(peso, valori, modifiche)
                    writer.writerow({
                        "riga_csv": riga.numero, "years": riga.anno, "modalita": modo,
                        "segmento": indice, "token_originali": peso, **campi_punteggi(valori),
                        "sostituzioni": json.dumps(modifiche, ensure_ascii=False, sort_keys=True),
                        "testo_originale": varianti["originale"], "testo_analizzato": testo,
                    })
            totale_segmenti += len(blocco)
            if totale_segmenti % (args.batch_size * 25) == 0:
                print(f"Segmenti elaborati: {totale_segmenti:,}", flush=True)
    return medie


def scrivi_righe(righe, modi, medie, cartella):
    annuali = defaultdict(Counter)
    campi = ["riga_csv", "years", "modalita", "stato", "segmenti", "token_originali",
             *ETICHETTE, *COLONNE_SCORE, "sostituzioni", "transcription"]
    with (cartella / "risultati_righe.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = scrittore(file, campi)
        for riga in righe:
            for modo in modi:
                conteggi = annuali[riga.anno, modo]
                conteggi["righe_csv"] += 1
                dati = {"riga_csv": riga.numero, "years": riga.anno, "modalita": modo, "transcription": riga.testo}
                media = medie.get((riga.numero, modo))
                if media is None:
                    conteggi["righe_senza_testo"] += 1
                    writer.writerow({**dati, "stato": "trascrizione_vuota", "segmenti": 0, "token_originali": 0})
                    continue

                valori = campi_punteggi({classe: totale / media.token for classe, totale in media.totali.items()})
                conteggi.update({
                    "righe_analizzate": 1, "segmenti": media.segmenti,
                    "sostituzioni": sum(media.sostituzioni.values()),
                    valori["sentiment"]: 1, valori["emozione"]: 1,
                })
                writer.writerow({
                    **dati, "stato": "analizzata", "segmenti": media.segmenti, "token_originali": media.token,
                    **valori, "sostituzioni": json.dumps(media.sostituzioni, ensure_ascii=False, sort_keys=True),
                })
    return annuali


def scrivi_anni(anni, modi, annuali, cartella):
    metriche = ["righe_csv", "righe_analizzate", "righe_senza_testo", "segmenti", "sostituzioni"]
    campi = ["years", "modalita", *metriche, *[f"n_{c}" for c in CLASSI], *[f"pct_{c}" for c in CLASSI]]
    with (cartella / "riepilogo_annuale.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = scrittore(file, campi)
        for anno in anni:
            for modo in modi:
                dati = annuali[anno, modo]
                n = dati["righe_analizzate"]
                writer.writerow({
                    "years": anno, "modalita": modo,
                    **{metrica: dati[metrica] for metrica in metriche},
                    **{f"n_{c}": dati[c] for c in CLASSI},
                    **{f"pct_{c}": round(dati[c] * 100 / n, 6) if n else None for c in CLASSI},
                })


def grafici(anni, modi, annuali, cartella):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    matplotlib.rcParams["svg.fonttype"] = "none"
    for tipo, classi in ETICHETTE.items():
        fig, assi = plt.subplots(
            len(modi), 1, squeeze=False, sharex=True, layout="constrained",
            figsize=(max(6.4, min(13, 2 + len(anni) * 0.75)), 1.1 + len(modi) * 3),
        )
        fig.get_layout_engine().set(w_pad=0.10, h_pad=0.10)
        titolo = "Sentiment per anno" if tipo == "sentiment" else "Emozioni per anno"
        fig.suptitle(titolo, fontsize=16, fontweight="bold", color="#213444")
        for ax, modo in zip(assi[:, 0], modi):
            conteggi = [annuali[anno, modo] for anno in anni]
            numerosita = [dato["righe_analizzate"] for dato in conteggi]
            base = [0.0] * len(anni)
            for classe in classi:
                quote = [dato[classe] * 100 / n if n else 0 for dato, n in zip(conteggi, numerosita)]
                ax.bar(anni, quote, width=0.7, bottom=base, color=COLORI[classe], label=NOMI[classe])
                base = [a + b for a, b in zip(base, quote)]
            if len(anni) <= 20:
                for anno, n in zip(anni, numerosita):
                    ax.text(anno, 102, f"n={n}", ha="center", va="bottom", fontsize=8, color="#455864")
                ax.set_xticks(anni)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
            ax.set_title(MODI[modo], loc="left", fontsize=11, color="#213444")
            ax.set_xlim(anni[0] - 0.8, anni[-1] + 0.8)
            ax.set_ylim(0, 113)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DEE5E9", linewidth=0.8)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#DEE5E9")
            ax.tick_params(length=0, colors="#455864")
        assi[-1, 0].set_xlabel("Anno")
        fig.supylabel("Quota di righe analizzate")
        handles, labels = assi[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=len(classi), frameon=False)
        for formato in ("svg", "png"):
            fig.savefig(cartella / f"{tipo}_per_anno.{formato}", dpi=240, facecolor="white")
        plt.close(fig)


def argomenti():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV con le colonne years e transcription")
    parser.add_argument("--da", type=int, help="primo anno incluso, o singolo anno se usato da solo")
    parser.add_argument("--a", type=int, help="ultimo anno incluso")
    parser.add_argument("--elenca-anni", action="store_true", help="mostra gli anni disponibili ed esce")
    parser.add_argument("--modalita", choices=[*MODI, "confronto"], default="storica", help="default: storica")
    parser.add_argument("--lessico", type=Path, default=LESSICO_PREDEFINITO, help="CSV con le regole di normalizzazione")
    parser.add_argument("--out", type=Path, help="nuova cartella in cui scrivere i risultati")
    parser.add_argument("--separatore", choices=["auto", ",", ";", "tab"], default="auto", metavar="SEP",
                        help="auto, virgola (,), punto e virgola (;) o tab")
    parser.add_argument("--codifica", default="utf-8-sig", help="codifica del CSV di ingresso, es. cp1252")
    parser.add_argument("--token-segmento", type=int, default=128, help="obiettivo di token per segmento (default: 128)")
    parser.add_argument("--batch-size", type=int, default=16, help="dimensione dei blocchi di inferenza (default: 16)")
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def scegli_anni(args, disponibili):
    if args.da is None and args.a is None:
        primo = int(input(f"Primo anno [{min(disponibili)}]: ").strip() or min(disponibili))
        ultimo = int(input(f"Ultimo anno [{primo}]: ").strip() or primo)
    else:
        primo = args.da if args.da is not None else args.a
        ultimo = args.a if args.a is not None else primo
    if not 1000 <= primo <= ultimo <= 9999:
        raise ValueError("Indica due anni di quattro cifre in ordine cronologico, oppure un solo anno.")
    return range(primo, ultimo + 1)


def main():
    args = argomenti()
    separatore = "\t" if args.separatore == "tab" else args.separatore
    righe, anni_invalidi, separatore = leggi_csv(args.csv, args.codifica, separatore)
    disponibili = Counter(riga.anno for riga in righe)
    if anni_invalidi:
        print(f"Record esclusi perché 'years' è vuoto o non valido: {len(anni_invalidi)}.", file=sys.stderr)
    if not disponibili:
        raise ValueError("Il CSV non contiene anni validi nella colonna 'years'.")
    if args.elenca_anni or (args.da is None and args.a is None):
        con_testo = Counter(riga.anno for riga in righe if riga.testo.strip())
        print("Anno    Righe con testo / Righe totali")
        for anno, n in sorted(disponibili.items()):
            print(f"{anno}    {con_testo[anno]} / {n}")
    if args.elenca_anni:
        return

    anni = scegli_anni(args, disponibili)
    selezionate = [riga for riga in righe if riga.anno in anni]
    analizzabili = sum(bool(riga.testo.strip()) for riga in selezionate)
    if not analizzabili:
        raise ValueError(f"Nessuna trascrizione non vuota fra {anni[0]} e {anni[-1]}.")
    if not 16 <= args.token_segmento <= LIMITE_MODELLO - 2 or args.batch_size < 1:
        raise ValueError("Usa --token-segmento fra 16 e 498 e --batch-size maggiore di zero.")

    modi = list(MODI) if args.modalita == "confronto" else [args.modalita]
    destinazione = (args.out or Path(f"feelit_{anni[0]}_{anni[-1]}_{args.modalita}")).resolve()
    if destinazione.exists():
        raise ValueError(f"La destinazione esiste già: {destinazione}. Scegli una nuova cartella con --out.")
    dati_lessico = args.lessico.read_bytes() if "storica" in modi else None
    regole, schema = carica_lessico(dati_lessico) if dati_lessico is not None else ({}, None)
    modelli, dispositivo = carica_modelli(args.dispositivo)
    print(f"Intervallo {anni[0]}–{anni[-1]}. Righe con testo: {analizzabili} su {len(selezionate)}.", flush=True)

    destinazione.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".feelit-", dir=destinazione.parent) as temporanea:
        cartella = Path(temporanea) / "risultati"
        cartella.mkdir()
        medie = analizza(selezionate, modi, modelli, regole, schema, args, cartella)
        annuali = scrivi_righe(selezionate, modi, medie, cartella)
        scrivi_anni(anni, modi, annuali, cartella)
        grafici(anni, modi, annuali, cartella)
        if dati_lessico is not None:
            (cartella / "lessico_usato.csv").write_bytes(dati_lessico)
        metadati = {
            "csv": str(args.csv.resolve()), "anni_inclusi": [anni[0], anni[-1]], "modalita": modi,
            "token_segmento": args.token_segmento, "batch_size": args.batch_size, "dispositivo": dispositivo,
            "righe_selezionate": len(selezionate), "righe_analizzate": analizzabili,
            "record_con_anno_non_valido": anni_invalidi, "codifica": args.codifica, "separatore": separatore,
            "versioni": {pacchetto: version(pacchetto) for pacchetto in ("feel-it", "torch", "transformers", "matplotlib")},
            "modelli": {
                tipo: {"nome": cl.model.name_or_path, "revisione": cl.model.config._commit_hash}
                for tipo, cl in modelli.items()
            },
        }
        (cartella / "metadati.json").write_text(json.dumps(metadati, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cartella.rename(destinazione)
    print(f"Risultati salvati in {destinazione}")


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        sys.exit("Errore. Indica l'intervallo con --da e --a quando l'input interattivo non è disponibile.")
    except ImportError as errore:
        sys.exit(f"Errore. Installa i pacchetti con 'python -m pip install -r requirements.txt'. {errore}")
    except (ValueError, OSError, csv.Error, RuntimeError, LookupError) as errore:
        sys.exit(f"Errore. {errore}")
