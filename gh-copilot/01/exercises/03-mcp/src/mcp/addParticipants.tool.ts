import z from 'zod/v3'
import { getPayload } from 'payload'
import config from '@payload-config'

const addParticipants = {
  name: 'addParticipants',
  description: 'Adds multiple participants to the Participants collection in Payload CMS',
  handler: async (args: Record<string, unknown>) => {
    const payload = await getPayload({ config })
    const participants = args.participants as Array<{
      nome: string
      cognome: string
      email: string
      gruppo: string
    }>
    const results = []
    for (const participant of participants) {
      const res = await payload.create({
        collection: 'participants',
        data: {
          nome: participant.nome,
          cognome: participant.cognome,
          email: participant.email,
          gruppo: participant.gruppo as 'Gruppo1' | 'Gruppo2',
        },
      })
      results.push(res)
    }

    return {
      content: [
        {
          type: 'text' as const,
          text: `Inserted participants: ${results.length}.`,
        },
      ],
    }
  },
  parameters: z.object({
    participants: z
      .array(
        z.object({
          nome: z.string().describe('First name of the participant'),
          cognome: z.string().describe('Last name of the participant'),
          email: z
            .string()
            .email()
            .describe('Email of the participant in the form n.cognome@websolute.it'),
          gruppo: z
            .string()
            .describe('Group of the participant. Either Gruppo1 or Gruppo2 randomly assigned'),
        }),
      )
      .describe('Array of participants to add'),
  }).shape,
}

export default addParticipants
