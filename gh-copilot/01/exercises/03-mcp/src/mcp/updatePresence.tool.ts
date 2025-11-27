import z from 'zod/v3'
import { getPayload } from 'payload'
import config from '@payload-config'

const updatePresence = {
  name: 'updatePresence',
  description:
    'Updates the presence status of multiple participants in the Participants collection in Payload CMS',
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
      const res = await payload.update({
        collection: 'participants',
        where: { email: { equals: participant.email } },
        data: {
          presenze: {
            [args.week as string]: true,
          },
        },
      })
      results.push(res)
    }

    return {
      content: [
        {
          type: 'text' as const,
          text: `Updated ${results.length} for week ${args.week}.`,
        },
      ],
    }
  },
  parameters: z.object({
    participants: z
      .array(
        z.object({
          email: z
            .string()
            .email()
            .describe('Email of the participant in the form n.cognome@websolute.it'),
        }),
      )
      .describe('Array of participants to add'),
    week: z
      .string()
      .describe(
        'Week to update the presence for, in the format YYYY-Www (e.g., 2025-W42, 2026-W1)',
      ),
  }).shape,
}

export default updatePresence
