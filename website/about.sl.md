---
title: O projektu
template: about.html
subtitle: Motivacija, poudarek in podatki projekta SLM4IE.
description: Zakaj SLM4IE obstaja, na kaj se raziskava osredotoča in kateri so ključni podatki projekta.
hide:
  - navigation
facts:
  - label: Naslov
    value: Majhni jezikovni modeli za brezkontekstno pridobivanje informacij v evropskih jezikih
  - label: Številka projekta
    value: <a href="https://cris.cobiss.net/ecris/si/sl/project/24346" target="_blank">Z2-70067</a>
  - label: Shema
    value: Podoktorski raziskovalni projekt ARIS
  - label: Financer
    value: <a href="https://www.aris-rs.si/" target="_blank">ARIS</a>, Javna agencija za znanstvenoraziskovalno in inovacijsko dejavnost Republike Slovenije
  - label: Gostitelj
    value: <a href="https://ailab.ijs.si/" target="_blank">Odsek za umetno inteligenco</a>, <a href="https://www.ijs.si/" target="_blank">Institut Jožef Stefan</a>
  - label: Partner
    value: <a href="https://eventregistry.org/" target="_blank">Event Registry</a>
  - label: Trajanje
    value: marec 2026 – februar 2028
  - label: Vodja projekta
    value: <a href="https://cris.cobiss.net/ecris/si/sl/researcher/50358" target="_blank">dr. Erik Novak</a>
  - label: Kontakt
    value: <a href="mailto:erik.novak@ijs.si">erik.novak@ijs.si</a>
---

## Motivacija

Projekt SLM4IE izhaja iz treh omejitev današnjih velikih jezikovnih modelov: zasebno delovanje je drago, besedila, ki so najbolj pomembna, nikoli niso bila vključena v njihove podatke za usposabljanje, prav tako pa na isto vprašanje ne odgovorijo dvakrat enako.

Občutljive vsebine (npr. medicinske, finančne, pravne) ni mogoče pošiljati v lastniške oblačne storitve, zato mora model delovati na strojni opremi v lasti organizacije, ta oprema pa stane več, kot si večina manjših ustanov lahko privošči. Istega gradiva ni na spletu, zato ga ni niti v podatkih za usposabljanje, zaradi česar se občutek modela za jezik oddaljuje od izrazoslovja in oblikovanja področja, ki naj bi ga bralo. Jeziki z omejenimi viri, med katerimi je tudi slovenščina, so v teh podatkih za usposabljanje prav iz tega razloga slabo zastopani.

Splošnost ima svojo ceno. Širok model plačuje svojo raznovrstnost pri vsaki, še tako majhni nalogi, v obliki procesorske moči, energije in stroškov. In ker model generira besedilo, lahko isto vprašanje ob naslednjem zagonu vrne drugačen odgovor. Ekstrakcija informacij pa zahteva ravno nasprotno: enak odgovor vsakič znova in od modela, ki ustreza stroju v prostoru.

## Usmeritev projekta

Projekt gradi majhne jezikovne modele za brezkontekstno pridobivanje informacij, kar pomeni brez označenih primerov in brez vsakokratnega usposabljanja za posamezno nalogo, in so hkrati dovolj majhni, da delujejo na komercialnem grafičnem pospeševalniku (GPU). Preizkušajo se tako arhitekture, ki temeljijo izključno na kodirnikih (encoder-only), kot tiste, ki temeljijo izključno na dekodirnikih (decoder-only), pri čemer se merita učinkovitost stiskanja in optimizacijskih tehnik v primerjavi s ceno natančnosti. Poleg modelov projekt ureja tudi referenčne nabore podatkov (benchmarke) za področja, ki jih obstoječi nabori ne zajemajo, kar omogoča neposredno primerjavo majhnega modela z dosti večjim pod enakimi pogoji. [Delovni sklopi](work-packages.md) opisujejo organizacijo dela in predvidena poročila.

Rezultati so objavljeni sproti in ne šele ob zaključku: modeli in nabori podatkov
na [Hugging Face](https://huggingface.co/eriknovak), podatkovni cevovod ter
koda za usposabljanje in evalvacijo pa [na GitHubu](https://github.com/eriknovak/SLM4IE),
vsak z ustrezno dokumentacijo, potrebno za uporabo. Vse je objavljeno v odprti kodi, kjer koli to dopuščajo licence.

## Podatki o projektu
