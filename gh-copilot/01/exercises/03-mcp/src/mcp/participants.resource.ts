const participantsResource = {
  name: 'participants',
  title: 'Course Participants',
  description: 'GitHub Copilot course participants',
  uri: 'course://participants.md',
  mimeType: 'text/markdown',
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handler: (uri: any) => ({
    contents: [
      {
        uri: uri.href,
        text: `# Course Participants

- Luca Marano
- Davide Mencarelli
- Mattia Fort
- Mirko Ugolini
- Riccardo Maugeri
- Cinzia Volponi
- Stefano Montini
- Roberto Piemonti
- Andrea Giorgini
- Fabio Di Giuseppe
- Aurel Gjeka
- Giacomo Gennari
- Stefano Tombari
- Stefania Riminucci
- Simone Palazzetti
- Luca Prosperi
- Giacomo Grassetti
- Carlo Munarini
- Francesco Basile
- Massimo Carletti
- Rocco Gallo
- Stefano Corinaldesi
- Luca Zampetti
- Luca De Blasio
- Manuel Pasini
- Andrea Armeli`,
      },
    ],
  }),
}

export default participantsResource
