;; ASCII ;;
; W: wall, D: door, k: key, g: gem, G: goal-gem, s: start, .: empty
; g..W...g
; k..D.Wk.
; .WWWWWWD
; .W..D.W.
; .W..W.W.
; ....W.W.
; WWW.W.W.
; s...W..G
(define (problem doors-keys-gems-3)
  (:domain doors-keys-gems)
  (:objects
    up down right left - direction
    key1 key2 - key gem1 gem2 gem3 - gem
  )
  (:init
    (= (xdiff up) 0) (= (ydiff up) 1)
    (= (xdiff down) 0) (= (ydiff down) -1)
    (= (xdiff right) 1) (= (ydiff right) 0)
    (= (xdiff left) -1) (= (ydiff left) 0)
    (= width 8) (= height 8)
    (= xpos 1) (= ypos 1)
    (wall 1 2) (wall 2 2) (wall 3 2)
    (wall 5 1) (wall 5 2) (wall 5 3) (wall 5 4)
    (wall 7 2) (wall 7 3) (wall 7 4) (wall 7 5) (wall 7 6)
    (wall 2 4) (wall 2 5) (wall 2 6)
    (wall 3 6) (wall 4 6) (wall 5 6) (wall 6 6) (wall 7 6)
    (wall 4 8) (wall 6 7)
    (door 4 7) (door 5 5) (door 8 6)
    (doorloc 4 7) (doorloc 5 5) (doorloc 8 6)
    (at key1 1 7) (at key2 7 7)
    (at gem1 1 8) (at gem2 8 8) (at gem3 8 1)
    (itemloc 1 7) (itemloc 7 7)
    (itemloc 1 8) (itemloc 8 8) (itemloc 8 1)
  )
  (:goal (has gem3))
)
