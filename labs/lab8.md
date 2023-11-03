# Lab 8

-------------------------

### Brystkreft

Lukas, når du sier:

*Behandling A fungerte dårlig i 37% og behandling B i 24% av tilfeller. Da er behandling B 37/24=1.54 gang bedre enn behandling A.*

Dette er litt misvisende, hva med å si at behandling B hadde 13 mindre dårlig utfall en behandling A. 

Bare fordi vi vet når en behandling var dårlig, betyr ikke at vi vet når den er bra. Det kan være for eksempel at modell A funker dårlig hos eldre, men værre hos yngre.
Det mangler litt data her, siden vi vet kun om behandlingen var utilfredstillende eller ikke, ikke hvis den var effektiv, eller til hvilken grad den var effektiv.

Lukas og Emma, konklusjonen av å bytte alle til behandling B er veldig risky. At 37% ikke likte behandling A, betyr fortsatt at det var 63% som ikke synes det var tilfredstillende. Det kan være dumt å bytte de over, siden det kan være de ikker liker den nye behandlingen, siden det var 24% som ikke likte den behandlingen. 
Det vil være lurt å bytte de som ikke likte behandling B over til A, og de som ikke likte A, over til B. Dett er for å se om de muligens hadde likt den andre behandlingen bedre. 
Hvis man vet at det er en del folk som ikke likte behandling B, hvorfor tvinge de til å fortsette med den?

Lukas du hadde et godt poeng, dere burde undersøke pasientgrupper i mer detalje, for å se hva som får pasienter til å ikke like behandlingene. Kanskje de som ikke likte A, heller ikke liker B? Kanskje man finner en variabel som forklarer hvorfor de ikke synes at noen av behandlingene var tilfredstillende. 

Emma har ett godt poeng, kanskje ikke forskjellen er kausal? Man kan bruke statistiske tester for å undersøke om det er sansynlig at så mange fra vær gruppe ikke likte behandlingen, men det er litt tricky å gjøre, siden hvilken av gruppene skal vi si er "sannheten". 

Ett annet eksempel på hvordan man kan mistolke dataen er å si:
I 63% av tilfeller ble pasientet med behandling A veldig tilfredstilt. Og, I 76% av tilfeller ble pasientet med behandling B veldig tilfredstilt. 

Som sakt lenger oppe, bare fordi vi vet når en behandling var dårlig, betyr ikke at vi vet når den er bra.

-------------------------

### Studie og lønn

Dette spørreskjemet må ta hensyn til personene den blir sent ut til, og må samle data på en effektiv, men etisk hensynsfull måte.

**Hvilket bachelor-studie tok du?:**
(Drop-down meny for alle bachelor studiene, la stå tom hvis ikke aktuelt)

Hvis du studerte en annen bachelor, gjerne skriv navnet på den her: [ ]


**Hvilket master-studie tok du?:**
(Drop-down meny for alle master studiene, la stå tom hvis ikke aktuelt) 

Hvis du studerte en annen master, gjerne skriv navnet på den her: [ ]

**Hvilket kjønn identifiserer du deg med?:**
Text box : [   ] 

(K=kvinne, M=Mann, gjerne skriv annet hvis det gjelder deg)

**Var du med i ett fagutvalg?:**

Hvis ja gjerne spesifiser hviket her : []

La stå tom hvis ikke dette er aktuelt

**Fikk du mange nye venner når du studerte?**

Ja/noen/ike egentlig

**Hva var gjennomsnitt-karakteren din på studie**

Drop down box fra A-E

La stå tom hvis ikke dette er aktuelt

**Hva var karakteren din på bachelor-oppgaven din?**

Drop down box fra A-E

**Hva var karakteren din på master-oppgaven din?**

Drop down box fra A-E

La stå tom hvis ikke dette er aktuelt

**Hva jobber du som?**

Dropdown om yrker (IT,utvikler,service... etc)

**Hvor lenge siden var det du sluttet å studere, og begynte å jobbe?**

Dropdown box for 0-10 år

**Hvor mye tjente du i året (før skatt) de siste 5 årene?**

År 1:
Dropdown for 0-10k, 10k-50k .. etc

År 2:
Dropdown for 0-10k, 10k-50k .. etc

etc....

År 5:
Dropdown for 0-10k, 10k-50k .. etc

--------------------


Man kan spør om bachelor og mastergrad (og hvilken bachelor/master) de har tatt, man kan også spør om hvilket kjønn de identifserer seg med. Det kan også være lurt å spør de spesfikt om hva de jobber innenfor, siden noen kan ha valgt en helt annen retning en informatikk. Hvis det var sånn at noen jobbet i kafe, og noen ble skamrik på egen buisness, vil dette føre til store outliers i dataen. 

Måten man spør om lønn burde enten være i månden eller pr år. Man må også spesifisere om dette er før eller etter skatt, siden dette kan skape mye variasjon i dataen.

Hvis man spør om personlighetstrekk er det litt vanskelig, siden det kan man umulig være objektiv på. Man kan ikke gå i dybden her, men man kan feks spør om noen har vært med i fagutvalg, eller om de fikk mange venner på studie. Det betyr ikke at de 100% er veldig utadvendt, men det kan hjelpe med å tyde dette frem. Det kan ha en bedre effekt en å bare spør etter om personen er utadvendt eller ikke. 

----------------
*Analyse*

En ting man må ta hensyn til er at, hvis vi vil ha data om hva folk tjener i løpet av de 5 første årene etter avsluttet utdanning, og man sender spørreundersøkelsen til folk som har jobbet i alt fra 10 til 1 år, vil man få mye mer data for hva man tjener 1 året, enn for det 5 året. Dette er ikke ett problem, men bare noe å tenke på hvis man skal analysere dataen seinere. 

Man må ta hensyn til kjønn når det kommer til analysedata, siden det kan være at det er flere av ett kjønn som studerer informatikk, og derfor vil det komme frem mer data om lønn og dette kjønnet. Eksempelvis, hvis man har data om 100 kvinner, og 10 menn, er det en høyere sannsynlighet at en av de kvinnene tjener veldig mye. Dette kan da føre dataen til å vise oss at kvinner "On-average" tjener bedre. 

En etisk problemstilling kan være at hvis dataen fra denne studien kommer ut, at det viser seg at noen studier tjener mye mer en andre, så kan folk bytte studie. Dette er jo sant, men kanskje noen bytter fra noe de liker til noe som tjener de mer penger. Det kan også være slikt at det vises seg gjennom dataen at menn som studerer studie A tjener mye mer en kvinner som studerer studie B. Dette er ikke direkte bevis på forskjellsbehandling, men kan åpne opp til en mulighet å finne ut mer. 

### Teste anbefalingssystem

Man må dele folk opp i 2 grupper, en gruppe med brukere som får den nye behandlingen, og en som ikke får den, som kontroll.

Måten man deler folk opp i, må skje på en etisk og lur måte. Hvis man deler folk opp i feks hvilket land de ser på Netflix fra, da vil resultatene være pgå at det er forskjeller i hva folk fra forskjellige land ser på.

Man må dele brukere opp på 



