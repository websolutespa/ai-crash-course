import (
	"fmt"
	"math"
)

func dist(a, b, c, d float64) float64 {
	x := a * math.Pi / 180
	y := c * math.Pi / 180
	z := (c - a) * math.Pi / 180
	w := (d - b) * math.Pi / 180
	p := math.Sin(z/2)*math.Sin(z/2) + math.Cos(x)*math.Cos(y)*math.Sin(w/2)*math.Sin(w/2)
	q := 2 * math.Atan2(math.Sqrt(p), math.Sqrt(1-p))
	return 6371 * q
}
