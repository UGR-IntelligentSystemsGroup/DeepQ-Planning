;; Dominio del juego Catapult
;; Subobjetivo (:goal) -> Ir a una catapulta sin pasar por ninguna otra o ir a la salida sin pasar por ninguna catapulta.

(define (domain Catapult)
  (:requirements :strips :typing :adl :negative-preconditions :equality)

  (:types
    Tree Water Player Catapult Exit - Locatable
    Cell
  )

  (:predicates
    (at ?l - Locatable ?c - Cell)
    (connected-up ?c1 ?c2 - Cell)
    (connected-down ?c1 ?c2 - Cell)
    (connected-left ?c1 ?c2 - Cell)
    (connected-right ?c1 ?c2 - Cell)
    (cannot-move ?p - Player) ;; After the player gets to a cell where there is a catapult or an exit, it cannot move anymore
  )

  (:action move-up
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
      (not (cannot-move ?p))	
      (at ?p ?c1)
      (connected-up ?c1 ?c2)

      (not (exists (?t - Tree) (at ?t ?c2)))
      (not (exists (?w - Water) (at ?w ?c2)))
    )
    :effect (and
      (not (at ?p ?c1))
      (at ?p ?c2)

      ;; The player can only get to a catapult or exit if that is the subgoal to achieve (since it cannot move afterwards)
      (when
        (or (exists (?cat - Catapult) (at ?cat ?c2))
        (exists (?e - Exit) (at ?e ?c2)))

        (cannot-move ?p)
      )
    )
  )

  (:action move-down
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
      (not (cannot-move ?p))	
      (at ?p ?c1)
      (connected-down ?c1 ?c2)

      (not (exists (?t - Tree) (at ?t ?c2)))
      (not (exists (?w - Water) (at ?w ?c2)))
    )
    :effect (and
      (not (at ?p ?c1))
      (at ?p ?c2)

      ;; The player can only get to a catapult or exit if that is the subgoal to achieve (since it cannot move afterwards)
      (when
        (or (exists (?cat - Catapult) (at ?cat ?c2))
        (exists (?e - Exit) (at ?e ?c2)))

        (cannot-move ?p)
      )
    )
  )

  (:action move-right
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
      (not (cannot-move ?p))	
      (at ?p ?c1)
      (connected-right ?c1 ?c2)

      (not (exists (?t - Tree) (at ?t ?c2)))
      (not (exists (?w - Water) (at ?w ?c2)))
    )
    :effect (and
      (not (at ?p ?c1))
      (at ?p ?c2)

      ;; The player can only get to a catapult or exit if that is the subgoal to achieve (since it cannot move afterwards)
      (when
        (or (exists (?cat - Catapult) (at ?cat ?c2))
        (exists (?e - Exit) (at ?e ?c2)))

        (cannot-move ?p)
      )
    )
  )

  (:action move-left
    :parameters (?p - Player ?c1 ?c2 - Cell)
    :precondition (and
      (not (cannot-move ?p))	
      (at ?p ?c1)
      (connected-left ?c1 ?c2)

      (not (exists (?t - Tree) (at ?t ?c2)))
      (not (exists (?w - Water) (at ?w ?c2)))
    )
    :effect (and
      (not (at ?p ?c1))
      (at ?p ?c2)

      ;; The player can only get to a catapult or exit if that is the subgoal to achieve (since it cannot move afterwards)
      (when
        (or (exists (?cat - Catapult) (at ?cat ?c2))
        (exists (?e - Exit) (at ?e ?c2)))

        (cannot-move ?p)
      )
    )
  )

)