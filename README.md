# SchapenInDeWei

Dit was een interessante opgave dat ik gekregen heb tijdens een interview.
Daarom dus dat ik het eens wou proberen te implementeren.

## De opgave:
Stel dat je een wei hebt met schapen die niet bewegen.
Wat is het kortst mogelijke hek dat je nodig hebt om ze hier allemaal binnen te zetten?

==> Strikvraag? Eigenlijk heb je toch geen hek nodig want de schapen bewegen niet... Dus ja... Die gaan toch niet lopen!


## Mijn oplossing:
De weg dat ik bewandeld heb is quasi-brute force:
- vind de meeste extreme X/Y-waarden
- bereken het centrum
- bereken alle afstanden vanuit het centrum
- verbind de verste punten met elkaar (en zie dat de lijnen elkaar niet kruisen
- nadien zag ik op de plot dat het niet echt optimaal was --> Dus ik heb er dan ook een optimizer ingestoken die (weeral brute-force) probeert om punten te verwijderen en checkt of alle coördinaten nog binnen deze nieuwe veelhoek liggen.


## Randopmerkingen:
- Waarschijnlijk zal hier wel een mooiere wiskundige oplossing voor zijn, maar ik wou geen google/stackoverflow ofzo gebruiken tijdens de implementatie (buiten het opzoeken van bepaalde wiskundige formules die al tig-jaar niet meer gebruikt zijn!).
- Het programma slaat de laatste seed op in "last_seed.dat". Deze kan je moven naar "use_seed.dat" om diezelfde seed te gebruiken (handig voor debug)
- Het programma genereert 2 afbeeldingen: "before_optimization.png" en "final_result.png". Redelijk vanzelfsprekend wat dat moet voorstellen :-)

