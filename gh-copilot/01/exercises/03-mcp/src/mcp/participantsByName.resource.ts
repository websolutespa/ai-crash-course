import { ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js'
import participants from './participants.json'

interface Participant {
  id: number
  name: string
  surname: string
  email: string
}

const participantsByNameResource = {
  name: 'participantsByName',
  title: 'Participants By Name',
  description: 'Access participants details by their name',
  uri: new ResourceTemplate('course://participants/{name}.json', { list: undefined }),
  mimeType: 'application/json',
  handler: async (uri: URL, { name }: { name: string }) => {
    const filtered = participants.filter(
      (p: Participant) => p.name.toLocaleLowerCase() === name.toLocaleLowerCase(),
    )

    if (filtered.length === 0) {
      return {
        error: {
          code: -32002,
        },
      }
    }

    return {
      contents: [
        {
          uri: uri.href,
          text: JSON.stringify(filtered, null, 2),
        },
      ],
    }
  },
}

export default participantsByNameResource
