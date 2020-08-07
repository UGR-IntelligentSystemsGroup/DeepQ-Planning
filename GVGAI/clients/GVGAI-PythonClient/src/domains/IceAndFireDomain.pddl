(define (domain IceAndFire)
  (:requirements :strips :typing :adl :negative-preconditions)
    
  (:types
    Avatar Tree Coin Trap Boots Ice Fire Exit - Locatable
    Cell
  )
    
  (:predicates
    (has-fire-boots)
    (has-ice-boots)
    (has-hazard ?c - Cell)
    (empty ?c - Cell)
    (non-traversable ?c - Cell)
    (at ?l - Locatable ?c - Cell)
    (connected-up ?c1 ?c2 - Cell)
    (connected-down ?c1 ?c2 - Cell)
    (connected-left ?c1 ?c2 - Cell)
    (connected-right ?c1 ?c2 - Cell)
    (cannot-move ?a - Avatar)
  )
 
  ;; Normal movement
  (:action move-up
    :parameters (?a - Avatar ?c1 ?c2 - Cell)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-up ?c1 ?c2)
      (not (has-hazard ?c2))
      (not (non-traversable ?c2))
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-down
    :parameters (?a - Avatar ?c1 ?c2 - Cell)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-down ?c1 ?c2)
      (not (has-hazard ?c2))
      (not (non-traversable ?c2))
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-left
    :parameters (?a - Avatar ?c1 ?c2 - Cell)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-left ?c1 ?c2)
      (not (has-hazard ?c2))
      (not (non-traversable ?c2))
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-right
    :parameters (?a - Avatar ?c1 ?c2 - Cell)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-right ?c1 ?c2)
      (not (has-hazard ?c2))
      (not (non-traversable ?c2))
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  ;; Movements using ice boots
  (:action move-up-ice
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?i - Ice)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-up ?c1 ?c2)
      (has-ice-boots)
      (has-hazard ?c2)
      (at ?i ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-down-ice
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?i - Ice)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-down ?c1 ?c2)
      (has-ice-boots)
      (has-hazard ?c2)
      (at ?i ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-left-ice
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?i - Ice)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-left ?c1 ?c2)
      (has-ice-boots)
      (has-hazard ?c2)
      (at ?i ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-right-ice
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?i - Ice)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-right ?c1 ?c2)
      (has-ice-boots)
      (has-hazard ?c2)
      (at ?i ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  ;; Movements using fire boots
  (:action move-up-fire
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?f - Fire)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-up ?c1 ?c2)
      (has-fire-boots)
      (has-hazard ?c2)
      (at ?f ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-down-fire
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?f - Fire)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-down ?c1 ?c2)
      (has-fire-boots)
      (has-hazard ?c2)
      (at ?f ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-left-fire
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?f - Fire)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-left ?c1 ?c2)
      (has-fire-boots)
      (has-hazard ?c2)
      (at ?f ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
  
  (:action move-right-fire
    :parameters (?a - Avatar ?c1 ?c2 - Cell ?f - Fire)
    :precondition (and
      (at ?a ?c1)
      (not (cannot-move ?a))
      (connected-right ?c1 ?c2)
      (has-fire-boots)
      (has-hazard ?c2)
      (at ?f ?c2)
    )
    :effect (and
      (not (at ?a ?c1))
      (at ?a ?c2)
      (when
        (exists (?e - Exit) (at ?e ?c2))
        (cannot-move ?a)
      )
    )
  )
)
