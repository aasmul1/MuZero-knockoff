# Models

Denne mappen inneholder forskjellige modeller som kan brukes sammen med (MCTS).

## Oversikt

- `super_model.py`: Definerer grensesnittet (interface) som alle modeller må følge**
- `perfect_model.py`: En modell som bruker de faktiske spillreglene for perfekt simulering
- `muzero_model.py`: En modell som bruker nevrale nettverk til å lære seg spilldynamikk

** dette er jo per def ikke en superklasse, men et interface. Så litt sykt å calle "super", men får man muligheten til å kalle en klasse/fil supermodell, ja da griper man den sjansen

## hva faen skjer egentlig? Jo..

Ideen her er å skille søkealgoritmen (MCTS) fra modellen som brukes til å simulere spillet. Dette gjør at vi kan bytte mellom forskjellige modeller uten å endre søkekoden.

En "perfekt" modell vet nøyaktig hvordan spillet fungerer. Den simulerer hva som skjer når du tar en handling og gir deg det nøyaktige resultatet.

MuZero-modellen er derimot litt mer fancy. Den prøver å lære seg hvordan spillet fungerer ved å bruke tre nevrale nettverk:
1. Et nettverk som lager en abstrakt representasjon av spilltilstanden
2. Et nettverk som forutsier hva som skjer når du tar en handling
3. Et nettverk som vurderer hvor bra en tilstand er

Det kule med MuZero er at den ikke trenger å vite spillreglene på forhånd - den lærer seg dem!

## Bruk

For å bruke en modell med MCTS, bare opprett en instans av modellen og send den til MCTS:

```python
# For perfekt modell
simulator = GameSimulator()  # dette må implementeres TODO!!
model = PerfectModel(simulator)
mcts = MCTS(model, num_simulations=800)

# For MuZero
repr_net = RepresentationNetwork()  # dette må implementeres TODO!!
dyn_net = DynamicsNetwork()  # dette må implementeres TODO!!
pred_net = PredictionNetwork()  # dette må implementeres TODO!!
model = MuZeroModel(repr_net, dyn_net, pred_net, action_space_size=4)
mcts = MCTS(model, num_simulations=50)